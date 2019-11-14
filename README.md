# Implicit Weight Uncertainty in Neural Networks
This repository contains the code for the paper Implicit Weight
Uncertainty in Neural Networks
([arXiv](https://arxiv.org/abs/1711.01297)).

There is a starting point of a reimplementation in Pytorch [here](https://github.com/pawni/BayesByHypernet_Pytorch).

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


### Commands to run experiments:
MNIST:
```
python run_bbh_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/bbh/ -x layer_a_prior1_fullkernel_noise8 --layer_wise_gen --noise_shape 8 -a --prior_scale 1. --full_kernel -c 0
python run_bbh_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/bbh/ -x layer_a_prior1_fullkernel_noise1 --layer_wise_gen --noise_shape 1 -a --prior_scale 1. --full_kernel -c 0
python run_bbh_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/bbh/ -x layer_a_prior1_fullkernel_noise64 --layer_wise_gen --noise_shape 64 -a --prior_scale 1. --full_kernel -c 0

python run_bbh_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/bbh/ -x layer_a_prior1_fullkernel_indnoise8 --independent_noise --layer_wise_gen --noise_shape 8 -a --prior_scale 1. --full_kernel -c 0
python run_bbh_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/bbh/ -x layer_a_prior1_fullkernel_indnoise1 --independent_noise --layer_wise_gen --noise_shape 1 -a --prior_scale 1. --full_kernel -c 0
python run_bbh_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/bbh/ -x layer_a_prior1_fullkernel_indnoise64 --independent_noise --layer_wise_gen --noise_shape 64 -a --prior_scale 1. --full_kernel -c 0

python run_dropout_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/dropout/ -x dropout_standard -c 0
python run_map_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/map/ -x map_standard -c 0
python run_ensemble_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/ensemble/ -x ensemble_standard -c 1
python run_mnf_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/mnf/ -x mnf_a -a -c 3
python run_bbb_exp.py -e 100 -p /vol/biomedic2/np716/bbh_uai/mnist/bbb/ -x prior_1 --prior_scale 1. -c 4



```

CIFAR:
```

python run_dropout_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/dropout/ -x dropout_standard -c 0
python run_map_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/map/ -x map_standard -c 0
python run_bbb_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/bbb/ -x prior_1 --prior_scale 1. -c 2
python run_ensemble_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/ensemble/ -x ensemble_standard -c 0
python run_mnf_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/mnf/ -x mnf_a -a -c 1
python run_bbh_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/bbh/ -x layer_a_prior1_fullkernel_noise8 --layer_wise_gen --noise_shape 8 -a --prior_scale 1. --full_kernel -c 1
python run_bbh_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/bbh/ -x layer_a_prior1_fullkernel_noise1 --layer_wise_gen --noise_shape 1 -a --prior_scale 1. --full_kernel -c 0
python run_bbh_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/bbh/ -x layer_a_prior1_fullkernel_noise64 --layer_wise_gen --noise_shape 64 -a --prior_scale 1. --full_kernel -c 0

python run_bbh_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/bbh/ -x layer_a_prior1_fullkernel_indnoise8 --independent_noise --layer_wise_gen --noise_shape 8 -a --prior_scale 1. --full_kernel -c 0
python run_bbh_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/bbh/ -x layer_a_prior1_fullkernel_indnoise1 --independent_noise --layer_wise_gen --noise_shape 1 -a --prior_scale 1. --full_kernel -c 0
python run_bbh_cifar_resnet_exp.py -e 200 -p /vol/biomedic2/np716/bbh_uai/cifar/bbh/ -x layer_a_prior1_fullkernel_indnoise64 --independent_noise --layer_wise_gen --noise_shape 64 -a --prior_scale 1. --full_kernel -c 0


```
