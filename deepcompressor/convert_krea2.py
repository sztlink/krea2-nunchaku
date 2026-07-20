r"""
Convert a deepcompressor SVDQuant W4A4 checkpoint of Krea 2 Turbo into the two-file Nunchaku format
consumed by ``nunchaku.models.transformers.transformer_krea2.NunchakuKrea2Transformer2DModel``.

Mirrors the FLUX driver in ``convert.py``. The 28 ``Krea2TransformerBlock`` heavy blocks are quantized;
q/k/v/gate are fused into one ``to_qkv_gate`` group (they share input, so the checkpoint stored one
smooth and one low-rank branch under ``attn.to_q``). Everything else in the block (norm_q, norm_k, norm1,
norm2, scale_shift_table) and every non-block module (img_in, time_embed, time_mod_proj, text_fusion,
txt_in, final_layer) is carried through unchanged into the unquantized file.

Usage:
    python -m deepcompressor.backend.nunchaku.convert_krea2 \
        --quant-path /path/to/run/model \        # holds model.pt, scale.pt, smooth.pt, branch.pt
        --output-root /path/to/out \
        --rank 32

``model.pt`` is regenerated on the build box from the base Krea 2 + the cached smooth/branch (minutes).
``smooth.pt`` / ``branch.pt`` are the two ``krea2-turbo.pt`` caches, renamed.
"""

import argparse
import json
import os

import safetensors.torch
import torch

from .convert import convert_to_nunchaku_transformer_block_state_dict, update_state_dict

# converted local name -> deepcompressor source linear(s) within the block
KREA2_LOCAL_NAME_MAP: dict[str, str | list[str]] = {
    "attn.to_qkv_gate": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate"],
    "attn.to_out.0": "attn.to_out.0",
    "ff.gate": "ff.gate",
    "ff.up": "ff.up",
    "ff.down": "ff.down",
}
# q/k/v/gate share to_q's input smooth; ff.gate shares ff.up's input smooth
KREA2_SMOOTH_NAME_MAP: dict[str, str] = {
    "attn.to_qkv_gate": "attn.to_q",
    "attn.to_out.0": "attn.to_out.0",
    "ff.gate": "ff.up",
    "ff.up": "ff.up",
    "ff.down": "ff.down",
}
# one branch for the whole qkv+gate group under to_q; ff.up/ff.gate/ff.down each have their own
KREA2_BRANCH_NAME_MAP: dict[str, str] = {
    "attn.to_qkv_gate": "attn.to_q",
    "attn.to_out.0": "attn.to_out.0",
    "ff.gate": "ff.gate",
    "ff.up": "ff.up",
    "ff.down": "ff.down",
}
KREA2_CONVERT_MAP: dict[str, str] = {
    "attn.to_qkv_gate": "linear",
    "attn.to_out.0": "linear",
    "ff.gate": "linear",
    "ff.up": "linear",
    "ff.down": "linear",
}

# source linear weights consumed by quantization (so we don't also copy them raw into `other`)
_QUANTIZED_LOCALS = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate", "attn.to_out.0", "ff.gate", "ff.up", "ff.down"]


def convert_to_nunchaku_krea2_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map=KREA2_LOCAL_NAME_MAP,
        smooth_name_map=KREA2_SMOOTH_NAME_MAP,
        branch_name_map=KREA2_BRANCH_NAME_MAP,
        convert_map=KREA2_CONVERT_MAP,
        float_point=float_point,
    )


def convert_to_nunchaku_krea2_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # discover block indices
    block_names: set[str] = set()
    for param_name in state_dict.keys():
        if param_name.startswith("transformer_blocks."):
            block_names.add(".".join(param_name.split(".")[:2]))
    block_names = sorted(block_names, key=lambda x: int(x.split(".")[-1]))
    print(f"Converting {len(block_names)} Krea2 transformer blocks...")

    # names of source weights the quantizer consumes, so they are not duplicated into `other`
    consumed: set[str] = set()
    for block_name in block_names:
        for local in _QUANTIZED_LOCALS:
            consumed.add(f"{block_name}.{local}.weight")
            consumed.add(f"{block_name}.{local}.bias")

    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        update_state_dict(
            converted,
            convert_to_nunchaku_krea2_block_state_dict(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
                block_name=block_name,
                float_point=float_point,
            ),
            prefix=block_name,
        )

    # everything the quantizer did not consume passes through unchanged (in-block norms and
    # scale_shift_table, plus img_in / time_embed / time_mod_proj / text_fusion / txt_in / final_layer)
    other: dict[str, torch.Tensor] = {
        name: param for name, param in state_dict.items() if name not in consumed
    }
    return converted, other


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, required=True, help="dir with model.pt, scale.pt, smooth.pt, branch.pt")
    parser.add_argument("--output-root", type=str, default="", help="output dir (default: quant-path)")
    parser.add_argument("--model-name", type=str, default="krea2-turbo-w4a4", help="output subdir name")
    parser.add_argument("--rank", type=int, default=32, help="low-rank branch rank used at calibration")
    parser.add_argument("--float-point", action="store_true", help="float-point 4-bit (nvfp4) instead of int4")
    args = parser.parse_args()
    output_root = args.output_root or args.quant_path

    map_location = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    state_dict = torch.load(os.path.join(args.quant_path, "model.pt"), map_location=map_location, weights_only=False)
    scale_dict = torch.load(os.path.join(args.quant_path, "scale.pt"), map_location="cpu", weights_only=False)
    smooth_path = os.path.join(args.quant_path, "smooth.pt")
    branch_path = os.path.join(args.quant_path, "branch.pt")
    smooth_dict = torch.load(smooth_path, map_location=map_location, weights_only=False) if os.path.exists(smooth_path) else {}
    branch_dict = torch.load(branch_path, map_location=map_location, weights_only=False) if os.path.exists(branch_path) else {}

    converted_state_dict, other_state_dict = convert_to_nunchaku_krea2_state_dicts(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        float_point=args.float_point,
    )

    output_dirpath = os.path.join(output_root, args.model_name)
    os.makedirs(output_dirpath, exist_ok=True)
    metadata = {"quantization_config": json.dumps({"rank": args.rank, "precision": "nvfp4" if args.float_point else "int4"})}
    safetensors.torch.save_file(
        converted_state_dict, os.path.join(output_dirpath, "transformer_blocks.safetensors"), metadata=metadata
    )
    safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
    print(f"Krea2 W4A4 Nunchaku checkpoint saved to {output_dirpath}")
    print(f"  quantized blocks: {len(converted_state_dict)} tensors; unquantized: {len(other_state_dict)} tensors")
