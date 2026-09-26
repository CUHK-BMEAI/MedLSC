# MedLSC

> **MedLSC: Learning to Specialize and Collaborate: Towards Hospital-centric Biomedical Multimodal Large Language Models That Learn Continually**

## Overview

Medical multimodal large language models (MLLMs) are increasingly expected to function as continually updated hospital-centric systems rather than static models. However, existing medical vision-language benchmarks mainly evaluate models under static protocols, while current MLLM continual-learning studies rarely reflect long-horizon, service-unit-organized hospital data streams and often assume known task identity during inference.

As a result, it remains unclear whether current MLLM continual-learning schemes can continually acquire, retain, and selectively reuse heterogeneous medical knowledge under realistic task-agnostic deployment.

## Contributions

### HoC-MedVL Benchmark

To bridge the gap between realistic deployment and existing benchmarks, we introduce **HoC-MedVL**, a hospital-centric continual-learning benchmark organized by clinical service units.

- **13** medical vision-language datasets
- **6** clinical service units
- **10** imaging modalities
- **3** task types: classification, visual question answering (VQA), and report generation

HoC-MedVL exposes a challenging setting where models must learn sequentially from diverse departments while performing inference **without oracle task or department identities**.

### MedLSC Framework

To address these challenges, we propose **MedLSC**, a progressive expert-based continual adaptation framework for hospital-centric MLLMs. MedLSC follows an *isolate–identify–cooperate* design:

1. **Isolate** — Expands dataset-specific LoRA experts to absorb new medical knowledge without overwriting previous expertise.
2. **Identify** — Consolidates a lightweight learnable expert-key router to identify relevant experts from the image-instruction input itself.
3. **Cooperate** — Introduces department-aware anchor-aided collaborative routing, which uses training-free task anchors to estimate department-level relevance and adaptively combines anchor-based priors with learned routing scores to activate a compact set of same- and cross-department experts.

Extensive experiments on HoC-MedVL show that existing continual-learning baselines struggle with forgetting and knowledge conflict, while MedLSC achieves stronger overall performance and more reliable long-horizon knowledge retention.

## Getting Started

*Code and data will be released soon. Stay tuned!*

## Results
The complete MedLSC experimental results are available here:
[Download MedLSC results](https://drive.google.com/file/d/102ScCRq2aHp3b6vBHv7wi4zUNGaj3cnO/view?usp=sharing)

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{medlsc2026,
  title     = {Learning to Specialize and Collaborate: Towards Hospital-centric Medical Multimodal Large Language Models That Learn Continually},
  author    = {},
  year      = {2026}
}
```
