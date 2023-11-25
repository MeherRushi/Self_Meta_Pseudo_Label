# Self_Meta_Pseudo_Label(SMPL)
![Progress](https://img.shields.io/badge/Progress-90%25-green)

## Overview
This repository contains the implementation of the Self Meta Pseudo Labels (SMPL) method for semi-supervised learning. SMPL is an extension of Meta Pseudo Labels, introducing a variant that reduces VRAM usage and a novel two-step gradient update process.

## Table of Contents
- [Background](#background)
- [Self Meta Pseudo Labels](#self-meta-pseudo-labels)
- [Implementation Details](#implementation-details)
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
> We have written code in a rudimentary format, we plan to further modularize it and add scripts to run it directly from terminal later
We plan to structure the directory as follows :
<!-- The implementation is structured as follows: -->
- `smpl_model.py`: Contains the implementation of the Self Meta Pseudo Labels model.
- `data_augmentation.py`: Implements various data augmentation policies such as Unsupervised Data Augmentation, AutoAugment, and RandAugment.
- `train.py`: Training script for the SMPL model.
- `evaluate.py`: Script for evaluating the trained model on a dataset.

## Experiments
We conducted experiments to validate the performance of SMPL. For details on the toy experiment, please refer to the [Experiments](#experiments) section in the [paper](https://arxiv.org/abs/2003.10580).

### Experiment with Moon Dataset

To better understand the performance of Self Meta Pseudo Labels (SMPL), we conducted a toy experiment on the moon dataset from Scikit-learn. The moon dataset is a simple toy dataset with 2D data points, consisting of two interleaving half circles on a 2D plane. Here are the details of the experiment:

### Dataset
- **Moon Dataset**: We generated a moon dataset with 2,000 examples divided into two classes. The dataset consists of 2D points with two interleaving half circles.
- **Labeling**: Six examples were randomly selected as labeled examples, and the remaining examples were used as unlabeled data.

### Model and Training Details
- **Model Architecture**: We employed a simple neural network with two fully connected hidden layers, each having 8 units. The activation function used was ReLU.
- **Training Configuration**: Data augmentations and regularization losses were removed for this experiment.
- **Learning Rate and Steps**: The model was trained with an initial learning rate of 0.1 for 1,000 steps.
- **Supervised Learning Comparison**: For the supervised learning experiment, the model was trained with the same hyperparameters, but using only the labeled examples.

### Illustrating the Gradient Descent Process
To explain our approach, we visualized the gradient descent process with an example. We projected the cost function to a 3D space, as shown in Figure 2. The red arrow represents the gradient descent of vanilla supervised learning, moving towards the global minimum. In SMPL, we introduced two gradient updates in each training epoch, depicted by two blue arrows. The first update moves away from the global minimum, while the second update corrects the direction, resulting in the final result being closer to the global minimum compared to vanilla supervised learning.

### Experiments performed

- Considered an equal distribution of the labeled data (3 in each class)
  - Exp-1 : We ran for the ground truth - We considered all the data to be labelled and trained the model and we got an accuracy of 95% which technically is the upper bound that can be achived with the given model and hyperparameters

  - Exp-2 : We now considered only the labeled data points , that is we trained the model on only 6 data points and achieved an accuracy of 76.5% which was expected because of very less data points

  - Exp-3 : We implemented the implementation of Meta Psuedo Labels paper to have a comparisoin standard and got a accuracy of 66% for the student model

  - Exp-4 : We finally implemented the Self-Meta-Psuedo-Label algorithm and we achieved a really good accuracy of 86%. 

- Considered an unequal distribution of the labeled data (5,1 split)
  - Exp-1 : We now considered only the labeled data points , that is we trained the model on only 6 data points and achieved an accuracy of 60% which was expected because of very less data points

  - Exp-2 : We implemented the implementation of Meta Psuedo Labels paper to have a comparisoin standard and got a accuracy of 77% for the student model

  - Exp-3 : We finally implemented the Self-Meta-Psuedo-Label algorithm and we achieved a really good accuracy of 86%. 

#### Please refer to the notebook for the diagramatic representation of the results.


### Results
We achieved the following accuracies in our experiments:
- **SMPL Accuracy**: 81.15%
- **Supervised Learning Accuracy**: 76.55%
- **Accuracy Improvement**: SMPL outperformed conventional supervised learning by 4.6% using the same model infrastructure. This improvement is attributed to SMPL's ability to leverage information from unlabeled data during updates.


### Experiment with CIFAR-10 Dataset
# Experiments on CIFAR-10-4K, CIFAR-100-10K, and SVHN-1K Datasets

## Datasets
We conducted experiments on three standard datasets:

- **CIFAR-10-4K:** 4,000 labeled images, 46,000 unlabeled images
- **CIFAR-100-10K:** 10,000 labeled images, 40,000 unlabeled images (100 classes)
- **SVHN-1K:** 1,000 labeled images, 603,000 unlabeled images (10 classes)

## Training Details
Our training procedure involves two stochastic gradient descent steps in every training epoch. In step one, we draw a batch of labeled and unlabeled data, generate pseudo labels using model predictions, and compute the gradient for the first objective function. In step two, we update the model based on semi-supervised and unsupervised loss functions. We clip the gradient norm at 0.8.


> We are currently training this model. But we are facing issues in replicating the hyperparameters of the paper as it proposes to train the model for 8000 epochs which is very difficult to replicate the results with limited infrastructure. 
<!-- After training, we fine-tune the best checkpoint on labeled data for improved accuracy. The finetuning process involves retraining the model with labeled data using stochastic gradient descent for 8,000 epochs with a fixed learning rate of 5e-6. -->

## Results
While we were unable to successfully re-run the Meta Pseudo Labels experiments with the official released code and instructions, we replicated our version of Meta Pseudo Labels using PyTorch on the CIFAR-10-4K dataset.
<!-- 
 We achieved an accuracy of 95.87% compared to 96.11% in the original paper. Additionally, we achieved an accuracy of 95.91% using Self Meta Pseudo Labels on the CIFAR-10-4K dataset with a 19.3% reduction in VRAM usage. -->

<!-- On the SVHN-1K dataset, we achieved 94.55% and 95.69% accuracy with the Meta Pseudo Labels method and our method, respectively, along with a 19.1% reduction in VRAM usage. We achieved an accuracy of 78.32% on the CIFAR-100-4K dataset. -->

<!-- Refer to the paper for detailed results, comparisons, and analysis. The provided implementation details serve as a starting point for running experiments and may be further improved for ease of use and modularity. -->


## Usage
<!-- To train the SMPL model, follow these steps:
1. Install dependencies (`requirements.txt`).
2. Prepare your labeled and unlabeled datasets.
3. Run the `train.py` script, specifying the dataset paths and hyperparameters.

Example:
```bash
python train.py --labeled_data_path path/to/labeled/data --unlabeled_data_path path/to/unlabeled/data --epochs 50 -->

> building the pipeline under progess

We are running the google colab notebooks. So to see our results, just open the notebooks for reference
All the cells can be run again using `T4 GPU` of google colab as the runtime.

## Results
In our experiments, SMPL demonstrated improved accuracy compared to conventional supervised learning. For detailed results, please refer to the Results section in the [paper](https://arxiv.org/abs/2003.10580).

## Dependencies
- Python 3.x
- Pytorch
- Other dependencies listed in `requirements.txt`
Install dependencies using:
```bash
pip install -r requirements.txt
```

     
