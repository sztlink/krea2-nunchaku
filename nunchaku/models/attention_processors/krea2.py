"""
Attention processor for :class:`~nunchaku.models.transformers.transformer_krea2.NunchakuKrea2Attention`.

Mirrors ``diffusers.models.transformers.transformer_krea2.Krea2AttnProcessor`` exactly, so the
numerics are identical to full precision. The only difference is that the q/k/v/gate/out projections
are :class:`~nunchaku.models.linear.SVDQW4A4Linear` (fused W4A4) instead of ``nn.Linear``; every other
op (grouped-query attention, zero-centered q/k RMSNorm, 3D rotary, the sigmoid output gate) runs eager
in bf16/fp32, unchanged from the reference. Kernel fusion of QKV+norm+rope and the gate as a 4th GEMM
output is a later optimization.
"""

from typing import Optional

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb


class NunchakuKrea2AttnProcessor:
    """W4A4 attention processor for Krea 2 single-stream blocks."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # One fused W4A4 GEMM for q/k/v/gate (they share the same input, one smooth + one low-rank branch,
        # matching how the checkpoint was calibrated), then split. GQA: q has num_heads, k/v num_kv_heads.
        qkv_gate = attn.to_qkv_gate(hidden_states)
        q_dim = attn.num_heads * attn.head_dim
        kv_dim = attn.num_kv_heads * attn.head_dim
        query, key, value, gate = torch.split(qkv_gate, [q_dim, kv_dim, kv_dim, attn.hidden_size], dim=-1)
        query = query.unflatten(-1, (attn.num_heads, attn.head_dim))
        key = key.unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = value.unflatten(-1, (attn.num_kv_heads, attn.head_dim))

        # Zero-centered RMSNorm on q/k (kept in fp32 by the source module).
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Expand the grouped-query k/v heads to full MHA instead of passing enable_gqa=True.
        # This is load-bearing, not cosmetic. Krea 2 always hands SDPA an attention mask, and the
        # flash backend does not serve enable_gqa together with a mask, so it falls back to the math
        # backend, materializes the full attention matrix, and costs ~3x end to end. A four-way
        # ablation on the real pipeline (expand x dtype-cast, 1024px 8 steps) isolated it:
        #   enable_gqa 22.3s | enable_gqa + cast 22.4s | expand 7.7s | expand + cast 7.8s
        # The expansion is what engages FlashAttention. The dtype cast makes no measurable difference
        # and is kept only for dtype hygiene after the fp32 q/k RMSNorm.
        if attn.num_heads != attn.num_kv_heads:
            n_rep = attn.num_heads // attn.num_kv_heads
            key = key.repeat_interleave(n_rep, dim=2)
            value = value.repeat_interleave(n_rep, dim=2)

        dtype = value.dtype
        query, key = query.to(dtype), key.to(dtype)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)
