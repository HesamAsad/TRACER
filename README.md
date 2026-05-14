<div align="center">

# TRACER

### Persistent Regularization for Robust Multimodal Finetuning

**Hesam Asadollahzadeh**, **Feng Liu**, **Christopher Leckie**, **Sarah M. Erfani** · *University of Melbourne* · **ICML&nbsp;2026** · [OpenReview](https://openreview.net/forum?id=XOYXLQRlj8)

*Official PyTorch code for **TRACER** — **T**rajectory-**R**obust **A**nchoring for **C**ontrastive **E**ncoder **R**egularization.*

*Corresponding author:* [h.asadollahzadeh@unimelb.edu.au](mailto:h.asadollahzadeh@unimelb.edu.au)

<br>

<img src="figure/figure_v6.jpg" alt="Overview of TRACER: contrastive finetuning with a WMA teacher and distillation." width="88%">

<br>

**Figure 1.** TRACER couples multimodal contrastive learning with self-distillation from a **weighted moving-average (WMA)** teacher trained along the trajectory.  
The student is optimized with **L<sub>MMCL</sub>**; the teacher supplies **L<sub>SD-WMA</sub>** to preserve orthogonal pretrained structure while adapting in the task subspace (see Algorithm 1 in the paper).

<br>

---

</div>

## Contents

- [Abstract](#abstract)
- [Main contributions](#main-contributions)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Quick start](#quick-start)

---

## Abstract

> Fine-tuning pretrained multimodal models improves in-distribution (ID) accuracy but often erodes **out-of-distribution (OOD) robustness**—a hallmark of catastrophic forgetting.
>
> We study contrastive fine-tuning through the **contrastive target matrix**, a reformulation that turns the linearized objective into a **matrix least-squares** problem and makes the geometry explicit: adaptation in the task subspace versus preservation along orthogonal directions.
>
> Classical **EMA teachers** progressively weaken their regularizing gap to the student. **Weighted moving-average (WMA) teachers** integrate the **optimization trajectory** and retain meaningful regularization over finite horizons. **TRACER** combines multimodal contrastive learning with **WMA-guided**, multi-perspective distillation. On CLIP fine-tuning, TRACER yields consistent **OOD accuracy and calibration** gains across architectures, backed by thorough **ablations** over distillation components, regularization strength, teacher update schedules, and kernel shape.

<p align="center"><strong>Keywords</strong><br><em>Multimodal contrastive learning · Fine-tuning · Robustness · Self-distillation</em></p>

---

## Main contributions

1. **Contrastive target matrix** — A least-squares view of linearized contrastive finetuning with closed-form insight into common recipes.
2. **Task vs. orthogonal geometry** — A decomposition that localizes forgetting and motivates dynamic teachers.
3. **Trajectory regularization** — We highlight **EMA collapse** of the teacher–student signal and show how **WMA** keeps a usable anchor; **TRACER** translates this into practice with strong empirical robustness gains.

---

## Quick start

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Adjust dataset paths inside the script first:
bash example_scripts/tracer.sh
```

```bash
python src/main.py --help
```

For all training flags, see [`src/args.py`](src/args.py).

---

## Repository layout

| Path | Role |
|:-----|:-----|
| [`src/models/tracer_loss.py`](src/models/tracer_loss.py) | TRACER loss and teacher / distillation logic |
| [`src/main.py`](src/main.py) | Training entry point |
| [`src/args.py`](src/args.py) | CLI / hyperparameters |
| [`example_scripts/tracer.sh`](example_scripts/tracer.sh) | Example launch script (edit paths for your machine) |

---

## Citation

If you use this code or the paper, please cite:

<details>
<summary><strong>BibTeX</strong> (click to expand)</summary>

```bibtex
@inproceedings{
asadollahzadeh2026tracer,
title={{TRACER}: Persistent Regularization for Robust Multimodal Finetuning},
author={Asadollahzadeh, Hesam and Liu, Feng and Leckie, Christopher and Erfani, Sarah M.},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=XOYXLQRlj8}
}
```

</details>

