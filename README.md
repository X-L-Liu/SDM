# Sequential Difference Maximization: Generating Adversarial Examples via Multi-Stage Optimization

> **Abstract:** Efficient adversarial attack methods are critical for assessing the robustness of computer vision 
> models. In this paper, we reconstruct the optimization objective for generating adversarial examples as "maximizing 
> the difference between the non-true labels' probability upper bound and the true label's probability," and propose 
> a gradient-based attack method termed **Sequential Difference Maximization** (**SDM**). SDM establishes a three-layer 
> optimization framework of "cycle-stage-step." The processes between cycles and between iterative steps are 
> respectively identical, while optimization stages differ in terms of loss functions: in the initial stage, the 
> negative probability of the true label is used as the loss function to compress the solution space; in subsequent 
> stages, we introduce the **Directional Probability Difference Ratio** (**DPDR**) loss function to gradually increase 
> the non-true labels' probability upper bound by compressing the irrelevant labels' probabilities. Experiments 
> demonstrate that compared with previous SOTA methods, SDM not only exhibits stronger attack performance but also 
> achieves higher attack cost-effectiveness. Additionally, SDM can be combined with adversarial training methods to 
> enhance their defensive effects.

## Installation

```
conda create -n SDM python=3.11.9
conda activate SDM
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
PyTorch is not version-sensitive. The project can typically run on other versions of PyTorch as well. 
Furthermore, allow the system to automatically select the version when installing any other missing libraries.

## Evaluate Model Robustness Using SDM

You can evaluate the computer vision models' robustness using methods such as "PGD," "C&W," "APGD-CE," "APGD-DLR," 
and "SDM."

The invocation method is as follows:  
  `python eval.py --dataset_name cifar10 --total_steps 20 --model_name WideResNet28x10 --model_path xxxx.pt`

## Enhance Adversarial Training (AT) Using SDM

You can improve the computer vision models' robustness using the SDM-based AT.

The invocation method is as follows:  
  `python train_AT.py --dataset_name cifar10 --attack SDM --model_name WideResNet28x10`

The code for other SDM-based defense methods will be released after the paper is accepted.