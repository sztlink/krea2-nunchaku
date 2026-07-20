# krea2-nunchaku

Krea 2 Turbo running on **4-bit weights and 4-bit activations** through Nunchaku's fused
low-bit kernels. This is the runtime port plus the deepcompressor conversion path, which is
everything you need to reproduce the build or run the checkpoint.

> The runtime port is [under review upstream](https://github.com/nunchaku-ai/nunchaku/pull/947).
> If it lands, install Nunchaku normally and skip the patching step below.

The checkpoint is at
[felipesztutman/Krea-2-Turbo-W4A4-Nunchaku](https://huggingface.co/felipesztutman/Krea-2-Turbo-W4A4-Nunchaku).
The fidelity and latency numbers, and the raw per-pair data behind them, are in
[dead-channel](https://github.com/sztlink/dead-channel).

Storage quantization (the fp8 and int8 Krea 2 builds in circulation) shrinks the file and
leaves the forward-pass arithmetic alone. This rewrites the arithmetic. It is 1.44x to 1.63x
over BF16 on an L40S, and it is the path that keeps going.

## What is here

| file | what it does |
|---|---|
| `nunchaku/models/transformers/transformer_krea2.py` | subclasses the diffusers `Krea2Transformer2DModel`, patches all 28 blocks' attention and feed-forward with `SVDQW4A4Linear` |
| `nunchaku/models/attention_processors/krea2.py` | attention processor, mirrors the diffusers one with the head-expansion fix below |
| `deepcompressor/convert_krea2.py` | maps deepcompressor's calibration output onto Nunchaku's key layout |
| `deepcompressor/krea2-turbo.yaml` | the model config used for calibration |

No C++ was needed. The existing Nunchaku kernels already serve this shape, so the port is
pure Python against them.

## Install

These files patch into installed copies of Nunchaku and deepcompressor.

```bash
git clone https://github.com/sztlink/krea2-nunchaku && cd krea2-nunchaku
NUNCHAKU=$(python -c "import nunchaku,os;print(os.path.dirname(nunchaku.__file__))")
cp nunchaku/models/transformers/transformer_krea2.py   "$NUNCHAKU/models/transformers/"
cp nunchaku/models/attention_processors/krea2.py       "$NUNCHAKU/models/attention_processors/"
```

Then register it in `$NUNCHAKU/models/transformers/__init__.py`:

```python
from .transformer_krea2 import NunchakuKrea2Transformer2DModel
```

Run it:

```python
from nunchaku.models.transformers.transformer_krea2 import NunchakuKrea2Transformer2DModel
from diffusers import Krea2Pipeline
import torch

transformer = NunchakuKrea2Transformer2DModel.from_pretrained(
    "felipesztutman/Krea-2-Turbo-W4A4-Nunchaku/svdq-int4_r32-krea-2-turbo.safetensors",
    torch_dtype=torch.bfloat16,
)
pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Turbo", transformer=transformer, torch_dtype=torch.bfloat16,
).to("cuda")

# Krea 2 Turbo is cfg-distilled. guidance_scale MUST be 0.0.
image = pipe("a fox in the snow", guidance_scale=0.0, num_inference_steps=8).images[0]
```

Ampere or newer. Verified on an RTX 3090 (sm_86): 2.66s at 512px/8 steps, 9.37s at 1024px/8
steps, 0.83s at 512px/2 steps, all warm medians. Also runs on Ada (4090, L40S, L4). Nunchaku
reports INT4 support down to Turing (20-series) since v1.2.0, which I have not tested here. The
NVFP4 variant is the one that needs Blackwell; this INT4 build does not.

## Two things that cost me time

**The GQA flag costs 3x when a mask is present.** Krea 2 uses grouped-query attention (48
query heads, 12 k/v heads) and always passes an attention mask, because text and image share
one sequence. PyTorch's SDPA will not serve `enable_gqa=True` together with a mask on the
flash backend. It falls back to the math backend silently and materializes the full attention
matrix. Expanding the 12 k/v heads to 48 by hand restores FlashAttention with the mask still
in place.

Four-way ablation on the real pipeline, 1024px, 8 steps:

| | time |
|---|---|
| `enable_gqa=True` | 22.3s |
| `enable_gqa=True` + dtype cast | 22.4s |
| expanded heads | 7.7s |
| expanded heads + dtype cast | 7.8s |

I got this wrong twice. First I blamed `enable_gqa`, then re-blamed an fp32 cast after an
isolated test seemed to clear the flag, then the full-pipeline ablation showed the first
answer was right. The isolated test had no mask, so it never met the condition that mattered.
Any model that splits heads this way and also masks will hit this, and will hit it silently.

**Fuse q/k/v/gate the way calibration grouped them.** deepcompressor smooths and builds the
low-rank branch over a fused projection, so the runtime has to fuse `to_q`, `to_k`, `to_v`
and `to_gate` into a single `to_qkv_gate` or the smoothing factors land on the wrong tensors.
The key naming also differs between the two projects (`lora_down`/`smooth` on one side,
`proj_down`/`smooth_factor` on the other), which `convert_krea2.py` handles.

## Reproducing the quantization

Calibrated on an H100 with deepcompressor, SVDQuant, rank 32, 128 calibration prompts. Copy
`deepcompressor/krea2-turbo.yaml` into `examples/diffusion/configs/model/` and
`convert_krea2.py` into `deepcompressor/backend/nunchaku/`, then run deepcompressor's
diffusion example against it and convert the output.

The rank-32 full-precision branch is what makes this hold. Calibration error climbs about 3x
from the first block to the last, and the branch absorbs the outliers that would otherwise
wreck the final blocks.

## License

The code here is MIT. The **checkpoint** it loads is a modified Krea 2 Turbo and is governed
by the Krea 2 Community License Agreement, not by this license. See the
[model repository](https://huggingface.co/felipesztutman/Krea-2-Turbo-W4A4-Nunchaku) for that.

Krea 2 is licensed under the Krea 2 Community License Agreement. For more information, visit
https://krea.ai/krea-2-licensing.

Built on [Nunchaku](https://github.com/nunchaku-tech/nunchaku) and
[deepcompressor](https://github.com/mit-han-lab/deepcompressor) from MIT Han Lab. SVDQuant is
theirs. This is a port of Krea 2 onto their runtime, not a new method.
