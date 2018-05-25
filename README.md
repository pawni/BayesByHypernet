# Implicit Weight Uncertainty in Neural Networks
This repository contains the code for the paper Implicit Weight
Uncertainty in Neural Networks
([arXiv](https://arxiv.org/abs/1711.01297)).

## Abstract
Modern neural networks tend to be overconfident on unseen, noisy or
incorrectly labelled data and do not produce meaningful uncertainty
measures. Bayesian deep learning aims to address this shortcoming with
variational approximations (such as Bayes by Backprop or Multiplicative
Normalising Flows). However, current approaches have limitations
regarding flexibility and scalability. We introduce Bayes by Hypernet
(BbH), a new method of variational approximation that interprets
hypernetworks as implicit distributions. It naturally uses neural
networks to model arbitrarily complex distributions and scales to
modern deep learning architectures. In our experiments, we demonstrate
that our method achieves competitive accuracies and predictive
uncertainties on MNIST and a CIFAR5 task, while being the most robust
against adversarial attacks.

## Usage
Following libraries were used for development:
```
future==0.16.0
jupyter==1.0.0
matplotlib==2.2.2
notebook==5.0.0
numpy==1.14.3
observations==0.1.4
pandas==0.19.2
scikit-learn==0.19.1
scipy==1.1.0
seaborn==0.8.1
tensorflow-gpu==1.7.0
tqdm==4.19.5
```

## Structure
`toy_data.ipynb` contains the code for the toy regression.
The other files contain the code for the mnist and cifar experiments.
`run_*` just calls the experiments. `base_layers` and `layers`
implement easy to use layers for different VI methods. `networks`
holds the models and the actual training and evaluation is in
`experiments` and `utils`.

## Contact
For discussion, suggestions or questions don't hesitate to
contact n.pawlowski16@imperial.ac.uk .
