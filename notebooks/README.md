# Notebooks — VibeThinker

This directory contains Jupyter notebooks for experimenting with VibeThinker models.

## Available Notebooks

| Notebook | Description | Platform |
|----------|-------------|----------|
| [`VibeThinker_3B_Inference_Colab.ipynb`](VibeThinker_3B_Inference_Colab.ipynb) | Colab-ready inference notebook — load model, run prompts, memory tips | Google Colab |
| [`vibethinker-inference.ipynb`](vibethinker-inference.ipynb) | Local starter notebook for running inference with transformers | Local |

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OMCHOKSI108/VibeThinkerModel/blob/main/notebooks/VibeThinker_3B_Inference_Colab.ipynb)

The Colab notebook covers:
- GPU runtime verification
- Dependency installation
- Optional Hugging Face login (via Colab secrets)
- Model loading from `OMCHOKSI108/VibeThinker-3B`
- Reusable inference function
- Example prompts (math, code, STEM)
- Memory optimization (4-bit/8-bit quantization)
- Troubleshooting and responsible use

## Prerequisites

All notebooks assume you have:
1. Python 3.10+
2. `transformers>=4.54.0` installed
3. A CUDA-capable GPU with sufficient VRAM (8 GB+ for bfloat16)
4. Model weights downloaded or accessible via Hugging Face

See [`docs/SETUP.md`](../docs/SETUP.md) for local setup and [`docs/INFERENCE.md`](../docs/INFERENCE.md) for inference parameters.

## Attribution

Model weights and original code belong to **WeiboAI and contributors**. See [`docs/ATTRIBUTION.md`](../docs/ATTRIBUTION.md) for full credits.
