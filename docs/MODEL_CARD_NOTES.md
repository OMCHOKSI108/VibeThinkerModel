# Model Card Notes

This document captures information from the original Hugging Face model card for reference.

## Original Model Card

The original model card for VibeThinker-3B is hosted at:
- https://huggingface.co/WeiboAI/VibeThinker-3B

A documented mirror is available at:
- https://huggingface.co/OMCHOKSI108/VibeThinker-3B

## Information from Original Sources

### Model Details

| Field | Value |
|-------|-------|
| Model Name | VibeThinker-3B |
| Parameters | 3 billion |
| Base Model | Qwen2.5-Coder-3B |
| Architecture | Dense Transformer |
| License | Not explicitly specified on the original model card |
| Pipeline Tag | text-generation |
| Library | transformers |

### Training Data

- Not specified in detail in the original source.
- The technical report ([arXiv:2606.16140](https://arxiv.org/pdf/2606.16140)) contains training details.

### Intended Use

- Mathematical reasoning
- Competitive programming
- STEM reasoning
- Instruction-following with explicit constraints
- Tasks with reliable verification signals

### Limitations (from original)

- For broad open-domain knowledge tasks, larger general-purpose models may still be more suitable.
- The model is optimized for verifiable reasoning tasks, not general-purpose chat.

## Notes on This Mirror

- The Hugging Face model at `OMCHOKSI108/VibeThinker-3B` is an unmodified mirror of the original weights.
- No additional training, fine-tuning, or quantization has been performed.
- Documentation files have been added to the model repository for clarity.
