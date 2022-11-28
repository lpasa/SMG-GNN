# Simple Multi-resolution Gated GNN

Pasa Luca, Nicolò Navarin, and Alessandro Sperduti.

Most Graph Neural Networks (GNNs) proposed in literature tend to add complexity (and non-linearity) to the model. In this paper, we follow the opposite direction by proposing a simple linear multi-resolution architecture that implements a multi-head gating mechanism. We assessed the performances of the proposed architecture on node classification tasks. To perform a fair comparison and present significant results, we re-implemented the competing methods from the literature and ran the experimental evaluation considering two different experimental settings with different model selection procedures. The proposed convolution, dubbed Simple Multi-resolution Gated GNN, exhibits state-of-the-art predictive performance on the considered benchmark datasets in terms of accuracy. In addition, it is way more efficient to compute than GAT, a well-known  multi-head GNN proposed in literature.

Paper: https://ieeexplore.ieee.org/abstract/document/9660046

If you find this code useful, please cite the following:

>@INPROCEEDINGS{Pasa2021SMGGNN,
>author={Pasa, Luca and Navarin, Nicolo and Sperduti, Alessandro}, 
>booktitle={2021 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
>title={Simple Multi-resolution Gated GNN}, 
>year={2021},
>pages={01-07},
>doi={10.1109/SSCI50451.2021.9660046}
>}

