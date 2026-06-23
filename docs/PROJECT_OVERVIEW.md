# Project Overview

## VibeThinker

VibeThinker is a family of dense reasoning models developed by **WeiboAI**, designed to explore how far verifiable reasoning can be pushed within small-model regimes.

### VibeThinker-3B

- **Parameters:** 3 billion
- **Base Model:** Qwen2.5-Coder-3B
- **Architecture:** Dense transformer
- **Post-Training Pipeline (upgraded SSP):**
  1. Curriculum-based supervised fine-tuning (SFT)
  2. Multi-domain reinforcement learning (RL)
  3. Offline self-distillation
  4. Instruction-oriented reinforcement learning
- **Target Tasks:** Mathematical reasoning, competitive programming, STEM reasoning, instruction-following with explicit constraints
- **Inference-Time Strategy:** Claim-Level Reliability Assessment (CLR) for test-time scaling
- **Technical Report:** [arXiv:2606.16140](https://arxiv.org/pdf/2606.16140)

### VibeThinker-1.5B

- **Parameters:** 1.5 billion
- **Base Model:** Not specified in the original source
- **Architecture:** Dense transformer
- **Post-Training Methodology:** Spectrum-to-Signal Principle (SSP)
  1. Two-Stage Diversity-Exploring Distillation (SFT phase)
  2. MaxEnt-Guided Policy Optimization (MGPO) (RL phase)
- **Target Tasks:** Competitive math and coding problems
- **Technical Report:** [arXiv:2511.06221](https://arxiv.org/abs/2511.06221)

## This Fork

This repository (`OMCHOKSI108/VibeThinkerModel`) is a **documented fork/mirror** of the original [WeiboAI/VibeThinker](https://github.com/WeiboAI/VibeThinker) GitHub repository. It exists to:

- Provide structured documentation (setup, inference, attribution)
- Serve as a learning resource for understanding the VibeThinker project
- Mirror the original model on Hugging Face with improved model cards
- Preserve all original code, license, and attribution

## Key Links

| Resource | Link |
|----------|------|
| Original GitHub | https://github.com/WeiboAI/VibeThinker |
| Original HF Model (3B) | https://huggingface.co/WeiboAI/VibeThinker-3B |
| Original HF Model (1.5B) | https://huggingface.co/WeiboAI/VibeThinker-1.5B |
| This Fork (GitHub) | https://github.com/OMCHOKSI108/VibeThinkerModel |
| This Fork (HF Mirror) | https://huggingface.co/OMCHOKSI108/VibeThinker-3B |

## News (from original)

- **[2026.06.16]** VibeThinker-3B released
- **[2025.11.19]** VibeThinker-1.5B hit #1 on Hugging Face trending models
- **[2025.11.11]** VibeThinker-1.5B open-sourced
- **[2025.11.05]** VibeThinker-1.5B announcement
