This is the PyTorch implementation for KEL proposed in the paper Knowledge-enhanced Graph Contrastive Learning for Recommendation

**SELFRec** is a Python framework for self-supervised recommendation (SSR) which integrates commonly used datasets and metrics, and implements many state-of-the-art SSR models. SELFRec has a lightweight architecture and provides user-friendly interfaces. It can facilitate model implementation and evaluation.

numba==0.53.1
numpy==1.20.3
scipy==1.6.2
tensorflow==1.14.0
torch>=1.7.0

How to run the codes:
1、python main.py
2、Select the model you want to run, like KECL
3、You can change the configuration of the parameters in the config file. Such as dataset, layers.

If you find this repo helpful to your research, please cite our paper.

@article{wang2025knowledge,
title={Knowledge-enhanced Contrastive Learning for Recommendation},
author={},
journal={},
year={2025}
}
