# Self_Meta_Pseudo_Label(SMPL)
![Progress](https://img.shields.io/badge/Progress-90%25-green)

## Overview
This repository contains the implementation of the Self Meta Pseudo Labels (SMPL) method for semi-supervised learning. SMPL is an extension of Meta Pseudo Labels, introducing a variant that reduces VRAM usage and a novel two-step gradient update process.

## Table of Contents
- [Background](#background)
- [Self Meta Pseudo Labels](#self-meta-pseudo-labels)
- [Implementation Details](#implementation-details)
  - [SMPL](#smpl-self-meta-pseudo-label)
  - [Colab_Notebooks](#colab-notebooks)
- [Experiments](#experiments)
  - [Experiment with Moon Dataset](#experiment-with-moon-dataset)
  - [Experiment with CIFAR-10 Dataset](#experiment-with-cifar-10-dataset)

- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)


## Background
Pseudo labeling is a semi-supervised approach for deep neural networks. This project builds upon the Meta Pseudo Labels method, addressing issues related to VRAM usage and the fixed teacher model during training. For a detailed background, please refer to the [original paper](https://arxiv.org/abs/2003.10580).

## Self Meta Pseudo Labels
SMPL introduces a variant of Meta Pseudo Labels that reduces VRAM usage by eliminating the need for an external teacher model. Instead, the model itself generates pseudo labels for unlabeled data and evaluates its performance through a two-step gradient update process.

## Implementation Details

We structure dthe directory as follows :

> It is recommended to run colab notebook for smoother access

### SMPL (self meta pseudo label)

To run the code, `clone the repository` and follow the below steps
```bash
$ cd SMPL
$ python3 master.py
```

The implementation is structured as follows:
- `data_handling.py`: Implements the creation of CIFAR-10 dataset and arrangement into proper file structure
- `data_transformation.py`: Implements various data augmentation policies such as Unsupervised Data Augmentation, AutoAugment, and RandAugment.
- `data_loading.py`: Functions to create the dataset and dataloader classes.
-  `model.py`: Contains the implementation of the Wide_resnet Self Meta Pseudo Labels.
- `train.py`: Training and evaluation functions for the SMPL model.
- `eval.py`: Evaluate the model   
-  `master.py` : Contains the main python script that is to be run 


### Colab Notebooks

> Just hit CTRL+F9 to run the notebooks (or Run all in the Top)

- **`Two_moon_exp.ipynb`** : This notebook contains our replication of the experiments regarding 2 moon dataset. We generated the datapoints randomly as mentioned in the paper. So the results may not be exactly the same but they replicate the conceptual requirement and show better performance in comparision to the other methods

- **`CIFAR_10_exp.ipynb`** : This notebook contains our replication of the experiments regarding CIFAR-10 dataset.
The major issue we faced is regarding the limitation of GPU, we only have access to the free T4 GPU from Colab. So we created a few free accounts and ran until we exhausted the GPU time and kept saving the model every 10 epochs. We managed to train the model for `72 epochs` and we got an `Test Accuracy` of `75.09%` and `Train accuracy` of `77.79%` which is better than a supervised model on 57 epochs. The paper requires us to train the model for **`8000`** epochs followed by fine-tuning which is `not` possible to replicate using the current resources at hand.

### Model Weights

This directory has the weights of the trained model

## Experiments

We conducted experiments to validate the performance of SMPL. For details on the toy experiment, please refer to the [Experiments](#experiments) section in the [paper](https://arxiv.org/abs/2003.10580).

### Experiment with Moon Dataset

To better understand the performance of Self Meta Pseudo Labels (SMPL), we conducted a toy experiment on the moon dataset from Scikit-learn. The moon dataset is a simple toy dataset with 2D data points, consisting of two interleaving half circles on a 2D plane. Here are the details of the experiment:

#### Dataset
- **Moon Dataset**: We generated a moon dataset with 2,000 examples divided into two classes. The dataset consists of 2D points with two interleaving half circles.
- **Labeling**: Six examples were randomly selected as labeled examples, and the remaining examples were used as unlabeled data.

#### Model and Training Details
- **Model Architecture**: We employed a simple neural network with two fully connected hidden layers, each having 8 units. The activation function used was ReLU.
- **Training Configuration**: Data augmentations and regularization losses were removed for this experiment.
- **Learning Rate and Steps**: The model was trained with an initial learning rate of 0.1 for 1,000 steps.
- **Supervised Learning Comparison**: For the supervised learning experiment, the model was trained with the same hyperparameters, but using only the labeled examples.

#### Illustrating the Gradient Descent Process

To explain our approach, we visualized the gradient descent process with an example. We projected the cost function to a 3D space, as shown in Figure 2. The red arrow represents the gradient descent of vanilla supervised learning, moving towards the global minimum. In SMPL, we introduced two gradient updates in each training epoch, depicted by two blue arrows. The first update moves away from the global minimum, while the second update corrects the direction, resulting in the final result being closer to the global minimum compared to vanilla supervised learning.

#### Experiments performed

- Considered an equal distribution of the labeled data (3 in each class)
  - Exp-1 : We ran for the ground truth - We considered all the data to be labelled and trained the model and we got an accuracy of 95% which technically is the upper bound that can be achived with the given model and hyperparameters

  - Exp-2 : We now considered only the labeled data points , that is we trained the model on only 6 data points and achieved an accuracy of 76.5% which was expected because of very less data points

  - Exp-3 : We implemented the implementation of Meta Psuedo Labels paper to have a comparisoin standard and got a accuracy of 66% for the student model

  - Exp-4 : We finally implemented the Self-Meta-Psuedo-Label algorithm and we achieved a really good accuracy of 86%. 

- Considered an unequal distribution of the labeled data (5,1 split)
  - Exp-1 : We now considered only the labeled data points , that is we trained the model on only 6 data points and achieved an accuracy of 60% which was expected because of very less data points

  - Exp-2 : We implemented the implementation of Meta Psuedo Labels paper to have a comparisoin standard and got a accuracy of 77% for the student model

  - Exp-3 : We finally implemented the Self-Meta-Psuedo-Label algorithm and we achieved a really good accuracy of 86%. 

##### Please refer to the notebook for the diagramatic representation of the results.


#### Results on the Two Moon Dataset

We achieved the following accuracies in our experiments:
- **SMPL Accuracy**: 81.15%
- **Supervised Learning Accuracy**: 76.55%
- **Accuracy Improvement**: SMPL outperformed conventional supervised learning by 4.6% using the same model infrastructure. This improvement is attributed to SMPL's ability to leverage information from unlabeled data during updates.


### Experiment with CIFAR-10 Dataset

#### Datasets
We conducted experiments on three standard datasets:

- **CIFAR-10-4K:** 4,000 labeled images, 46,000 unlabeled images
<!-- - **CIFAR-100-10K:** 10,000 labeled images, 40,000 unlabeled images (100 classes)
- **SVHN-1K:** 1,000 labeled images, 603,000 unlabeled images (10 classes) -->

## Training Details
Our training procedure involves two stochastic gradient descent steps in every training epoch. In step one, we draw a batch of labeled and unlabeled data, generate pseudo labels using model predictions, and compute the gradient for the first objective function. In step two, we update the model based on semi-supervised and unsupervised loss functions. We clip the gradient norm at 0.8.


> We are currently training this model. But we are facing issues in replicating the hyperparameters of the paper as it proposes to train the model for 8000 epochs which is very difficult to replicate the results with limited infrastructure. 
<!-- After training, we fine-tune the best checkpoint on labeled data for improved accuracy. The finetuning process involves retraining the model with labeled data using stochastic gradient descent for 8,000 epochs with a fixed learning rate of 5e-6. -->

We follow all the Hyperparameters given in the paper including Masking, label smoothening and Beta constant as well. The only hyperparameter that we could not replicate was training the model for 8000 epochs due to lack of computational resources.

All the issues and related work is documneted in [this pdf file](/public/smpl_rr.pdf)

## Results
While we were unable to successfully re-run the Meta Pseudo Labels experiments with the official released code and instructions, we replicated our version of Meta Pseudo Labels using PyTorch on the CIFAR-10-4K dataset.
<!-- 
 We achieved an accuracy of 95.87% compared to 96.11% in the original paper. Additionally, we achieved an accuracy of 95.91% using Self Meta Pseudo Labels on the CIFAR-10-4K dataset with a 19.3% reduction in VRAM usage. -->

<!-- On the SVHN-1K dataset, we achieved 94.55% and 95.69% accuracy with the Meta Pseudo Labels method and our method, respectively, along with a 19.1% reduction in VRAM usage. We achieved an accuracy of 78.32% on the CIFAR-100-4K dataset. -->

<!-- Refer to the paper for detailed results, comparisons, and analysis. The provided implementation details serve as a starting point for running experiments and may be further improved for ease of use and modularity. -->


## Results
In our experiments, SMPL demonstrated improved accuracy compared to conventional supervised learning. For detailed results, please refer to the Results on the CIFAR-10 Dataset section in the [this pdf file](/public/smpl_rr.pdf).


     
