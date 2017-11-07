# Implicit Weight Uncertainty in Neural Networks
This repository contains the code for the paper Implicit Weight Uncertainty in Neural Networks ([arXiv](https://arxiv.org/abs/1711.01297)).

## Abstract
We interpret HyperNetworks within the framework of variational inference within implicit distributions. Our method, Bayes by Hypernet, is able to model a richer variational distribution than previous methods. Experiments show that it achieves comparable predictive performance on the MNIST classification task while providing higher predictive uncertainties compared to MC-Dropout and regular maximum likelihood training.

## Usage
Following libraries were used for development:
```
future==0.16.0
jupyter==1.0.0
matplotlib==1.5.3
notebook==4.2.3
numpy==1.13.3
pandas==0.19.2
scipy==0.19.1
seaborn==0.7.1
tensorflow-gpu==1.4.0rc0
tqdm==4.11.2
```

## Structure
The notebooks contain the code for the two experiments. `toy_dataset.ipynb` contains the code for the toy regression. `MNIST.ipynb` contains the code for the MNIST digit classification.
## Contact
For discussion, suggestions or questions don't hesitate to contact n.pawlowski16@imperial.ac.uk .
