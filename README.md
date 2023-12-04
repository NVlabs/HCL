# Heterogeneous Continual Learning

Official PyTorch implementation of [**Heterogeneous Continual Learning**](https://arxiv.org/abs/2306.08593).

**Authors**: [Divyam Madaan](https://dmadaan.com/),  [Hongxu Yin](https://hongxu-yin.github.i), [Wonmin Byeon](https://wonmin-byeon.github.i), [Pavlo Molchanov](https://research.nvidia.com/person/pavlo-molchano),

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

---
**TL;DR: First continual learning approach in which the architecture continuously evolves with the data.**
-- 
## Abstract

![concept figure](https://github.com/divyam3897/cvpr_hcl/files/13549399/concept_figure.pdf)

We propose a novel framework and a solution to tackle
the continual learning (CL) problem with changing network
architectures. Most CL methods focus on adapting a single
architecture to a new task/class by modifying its weights.
However, with rapid progress in architecture design, the
problem of adapting existing solutions to novel architectures
becomes relevant. To address this limitation, we propose
Heterogeneous Continual Learning (HCL), where a wide
range of evolving network architectures emerge continually
together with novel data/tasks. As a solution, we build on
top of the distillation family of techniques and modify it
to a new setting where a weaker model takes the role of a
teacher; meanwhile, a new stronger architecture acts as a
student. Furthermore, we consider a setup of limited access
to previous data and propose Quick Deep Inversion (QDI) to
recover prior task visual features to support knowledge trans-
fer. QDI significantly reduces computational costs compared
to previous solutions and improves overall performance. In
summary, we propose a new setup for CL with a modified
knowledge distillation paradigm and design a quick data
inversion method to enhance distillation. Our evaluation
of various benchmarks shows a significant improvement on
accuracy in comparison to state-of-the-art methods over
various networks architectures.

__Contribution of this work__

- We propose a novel CL framework called Heteroge-
  neous Continual Learning (HCL) to learn a stream of
  different architectures on a sequence of tasks while
  transferring the knowledge from past representations.
- We revisit knowledge distillation and propose Quick
  Deep Inversion (QDI), which inverts the previous task
  parameters while interpolating the current task exam-
  ples with minimal additional cost.
- We benchmark existing state-of-the-art solutions in the
  new setting and outperform them with our proposed
  method across a diverse stream of architectures for both
  task-incremental and class-incremental CL.

## Prerequisites

```
$ pip install -r requirements.txt
```

## Quick start

### Training 

```python
python main.py --data_dir ../data/ --log_dir ./logs/scl/ -c configs/cifar10/distil.yaml --ckpt_dir ./checkpoints/c10/scl/distil/ --hide_progress --cl_default --validation --hcl

```

### Evaluation

```python
python linear_eval_alltasks.py --data_dir ../data/ --log_dir ./logs/scl/ -c configs/cifar10/distil.yaml --ckpt_dir ./checkpoints/c10/scl/distil/ --hide_progress --cl_default --hcl

```


To change the dataset and method, use the configuration files from `./configs`.

# Contributing

We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. If you have implementations of this repository in other ML frameworks, please reach out so we may highlight them here.

## Licenses

Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.


## Acknowledgment

The code is build upon [aimagelab/mammoth](https://github.com/aimagelab/mammoth), [divyam3897/UCL](https://github.com/divyam3897/UCL), [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/tree/master), [sutd-visual-computing-group/LS-KD-compatibility](https://github.com/sutd-visual-computing-group/LS-KD-compatibility), and [berniwal/swin-transformer-pytorch](https://github.com/berniwal/swin-transformer-pytorch).

## Citation

If you found the provided code useful, please cite our work.

```bibtex
@inproceedings{madaan2023heterogeneous,
  title={Heterogeneous Continual Learning},
  author={Madaan, Divyam and Yin, Hongxu and Byeon, Wonmin and Kautz, Jan and Molchanov, Pavlo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}

```
