# VibeThinkerModel

<p align="center"><img src="./figures/logo.png" width="100"/></p>

<p align="center">
  <a href="https://github.com/OMCHOKSI108/VibeThinkerModel">GitHub</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://huggingface.co/OMCHOKSI108/VibeThinker-3B">HF Mirror</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://github.com/WeiboAI/VibeThinker">Original GitHub</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="https://huggingface.co/WeiboAI/VibeThinker-3B">Original HF</a>
</p>

---

**This repository is a documented fork/mirror of [WeiboAI/VibeThinker](https://github.com/WeiboAI/VibeThinker) for learning, experimentation, and structured usage.**

It provides improved documentation, a structured project overview, setup guides, inference examples, and attribution. No model weights have been modified.

---

## What is VibeThinker?

VibeThinker is a family of dense reasoning models developed by WeiboAI to explore how far verifiable reasoning can be pushed in small-model regimes. Two variants are available:

| Model | Parameters | Base Model | Technical Report |
|-------|-----------|------------|-----------------|
| VibeThinker-3B | 3B | Qwen2.5-Coder-3B | [arXiv](https://arxiv.org/pdf/2606.16140) |
| VibeThinker-1.5B | 1.5B | Not specified in original source | [arXiv](https://arxiv.org/abs/2511.06221) |

VibeThinker-3B is post-trained with an upgraded Spectrum-to-Signal pipeline combining curriculum-based supervised fine-tuning, multi-domain reinforcement learning, offline self-distillation, and instruction-oriented reinforcement learning. It is designed for tasks with reliable verification signals including mathematical reasoning, competitive programming, STEM reasoning, and instruction-following with explicit constraints.

<p align="center"><img src="./figures/3b/Abstract.png" width="80%"/></p>

_VibeThinker-1.5B uses the "Spectrum-to-Signal Principle (SSP)" methodology with Two-Stage Diversity-Exploring Distillation and MaxEnt-Guided Policy Optimization (MGPO)._

<p align="center"><img src="./figures/1.5b/Abstract.png" width="80%"/></p>

## Benchmark Highlights

### VibeThinker-3B

<p align="center"><img src="./figures/3b/Acc_and_Scale.png" width="80%"/></p>

<p align="center"><img src="./figures/3b/VibeThiinker-3B.png" width="80%"/></p>

<p align="center"><img src="./figures/3b/VibeThinker-3B+CLR.png" width="80%"/></p>

<p align="center"><img src="./figures/3b/LeetCode.png" width="80%"/></p>

### VibeThinker-1.5B

<p align="center"><img src="./figures/1.5b/am25_1.5B.png" width="80%"/></p>

<p align="center"><img src="./figures/1.5b/Performence1.png" width="80%"/></p>

<p align="center"><img src="./figures/1.5b/Cost.png" width="80%"/></p>

### Training Architecture

<p align="center"><img src="./figures/3b/Architecture.png" width="80%"/></p>

<p align="center"><img src="./figures/1.5b/technicalArchitecture1.png" width="80%"/></p>

## Repository Structure

```
VibeThinkerModel/
├── README.md               # This file — fork overview
├── ORIGINAL_README.md      # Original README from WeiboAI (preserved verbatim)
├── README_old.md           # Original 1.5B-era README (preserved)
├── LICENSE                 # MIT License (original)
├── docs/
│   ├── PROJECT_OVERVIEW.md  # Detailed project description
│   ├── SETUP.md             # Environment setup and installation
│   ├── INFERENCE.md         # Inference examples and parameters
│   ├── MODEL_CARD_NOTES.md  # Notes on the Hugging Face model card
│   ├── ATTRIBUTION.md       # Credits and license information
│   └── ROADMAP.md           # Planned improvements to this fork
├── eval/                   # Evaluation scripts (original)
├── figures/                # Figures and assets (original)
├── VibeThinker-1.5B.pdf    # Original technical report PDF
├── VibeThinker-3B.pdf      # Original technical report PDF
└── .gitignore
```

## Quick Setup

```bash
# Clone this fork
git clone https://github.com/OMCHOKSI108/VibeThinkerModel.git
cd VibeThinkerModel

# Install dependencies
pip install transformers>=4.54.0
# Recommended for better performance:
# pip install vllm==0.10.1 sglang>=0.4.9.post6
```

See [SETUP.md](docs/SETUP.md) for detailed instructions.

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "WeiboAI/VibeThinker-3B",
    low_cpu_mem_usage=True,
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("WeiboAI/VibeThinker-3B", trust_remote_code=True)
```

See [INFERENCE.md](docs/INFERENCE.md) for full examples.

## Hugging Face Model Mirror

A documented mirror of the original Hugging Face model is available at:

- **Mirror:** [https://huggingface.co/OMCHOKSI108/VibeThinker-3B](https://huggingface.co/OMCHOKSI108/VibeThinker-3B)
- **Original:** [https://huggingface.co/WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B)

## License and Attribution

This code repository is licensed under the [MIT License](./LICENSE), inherited from the original repository. The original model/code credits belong to **WeiboAI and contributors**. See [ATTRIBUTION.md](docs/ATTRIBUTION.md) for full details.

## Disclaimer

**Original model/code credits belong to WeiboAI and contributors.** This fork is maintained for documentation, learning, and experimentation purposes. No claim of ownership or authorship of the original model is made.
