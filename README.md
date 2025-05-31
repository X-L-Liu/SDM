# Sequential Difference Maximization: Generating Adversarial Examples via Multi-Stage Optimization

[//]: # ([Paper]&#40;&#41; )

> **Abstract:** Efficient adversarial attack methods are critical for assessing the security of computer vision and 
> image information processing models. In this paper, we set the optimization objective as "maximizing the difference 
> between the non-true labels' probability upper bound and the true label's probability," and propose an adversarial 
> attack method named **Sequential Difference Maximization** (**SDM**). Based on the idea of sequential optimization, 
> this method establishes a three-layer optimization framework of "cycle-stage-step." The processes between cycles and 
> between iterative steps are respectively identical, while optimization stages differ in terms of loss functions: 
> the initial stage minimizes the true label's probability to compress the solution space, subsequent stages introduce 
> the **Directional Probability Difference Ratio** (**DPDR**) loss function to gradually increase the non-true labels' 
> probability upper bound. Experiments demonstrate that compared with previous SOTA methods, SDM not only exhibits 
> stronger attack performance but also achieves higher attack cost-effectiveness. Additionally, SDM can be combined 
> with adversarial training methods to enhance their defense effects.

## Installation

```
conda create -n SDM python=3.11.9
conda activate SDM
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
PyTorch is not version-sensitive. The project can typically run on other versions of PyTorch as well. 
Furthermore, allow the system to automatically select the version when installing any other missing libraries.

## Evaluate Models

You can evaluate the computer vision models using methods such as "PGD," "C&W," "APGD-CE," "APGD-DLR," and "SDM."

The invocation method is as follows:  
  `python eval.py --dataset_name cifar10 --total_steps 20 --model_name WideResNet28x10 --model_path xxxx.pt`
