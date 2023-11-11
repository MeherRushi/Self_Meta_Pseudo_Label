# Self_Meta_Pseudo_Label(SMPL)
![Progress](https://img.shields.io/badge/Progress-35%25-yellow)

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
The implementation is structured as follows:
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

### Results
We achieved the following accuracies in our experiments:
- **SMPL Accuracy**: 81.15%
- **Supervised Learning Accuracy**: 76.55%
- **Accuracy Improvement**: SMPL outperformed conventional supervised learning by 4.6% using the same model infrastructure. This improvement is attributed to SMPL's ability to leverage information from unlabeled data during updates.

### Visualization
For a visual representation of the results, refer to Figure 3.


### Experiment with CIFAR-10 Dataset

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

     
