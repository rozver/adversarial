# Black-box adversarial examples in the real world

Code for the paper "Black-box adversarial examples in the real world", authored by 
Hristo Todorov and Kristian Georgiev.

## Abstract
It has been shown that modern computer vision neural networks are vulnerable to adversarial examples -
altered inputs that are almost indistinguishable from natural data and yet classified incorrectly by the network.
In this paper, we create phyiscal adversarial examples using only a very limited number of queries to the targeted neural network.
We also execute both white-box and black-box attacks on various models. We analyze the features of the images that
effect the predictions of different types of models and demonstrate that it is possible to consistently create phyiscal
limited query adversarial examples that fool most of the state-of-the-art classifiers.

## Installation
~~~ 
git clone https://github.com/RoZvEr/adversarial.git
cd adversarial && pip install -r requirements.txt
python download_data.py
~~~

## Running experiments
#### PGD
Parameters:
~~~
python pgd.py
       [--arch {resnet18,resnet50,resnet152,alexnet,vgg16,vgg19,inception_v3}]
       [--dataset DATASET]
       [--masks]
       [--eps EPS]
       [--norm {l2,linf}]
       [--step_size STEP_SIZE]
       [--num_iterations NUM_ITERATIONS]
       [--targeted]
       [--eot]
       [--transfer]
       [--save_file_location SAVE_FILE_LOCATION]
~~~

Example usage:
~~~
python pgd.py
       --arch resnet50
       --dataset dataset/coco/airplane.pt
       --masks
       --eps 1.0
       --norm l2
       --step_size 1/255.0
       --num_iterations 50
       --eot
       --save_file_location results/example_pgd.pt
~~~

#### Model training (adversarial training featured)
Parameters:
~~~
python train.py
       [--arch {resnet18,resnet50,resnet152,alexnet,vgg16,vgg19,inception_v3}]
       [--dataset DATASET]
       [--pretrained]
       [--checkpoint_location CHECKPOINT_LOCATION]
       [--epochs EPOCHS] 
       [--learning_rate LEARNING_RATE] 
       [--adversarial] 
       [--save_file_location SAVE_FILE_LOCATION]
~~~

Example usage (adversarial training from scratch):
~~~
python train.py
       --arch resnet50
       --dataset dataset/imagenet-airplanes.pt
       --epochs 50
       --learning_rate 0.01
       --adversarial
       --save_file_location models/example_robust_model.pt
~~~

### Gradient analysis
Parameters:
~~~
python gradient_analysis.py
       [--arch {resnet18,resnet50,resnet152,alexnet,vgg16,vgg19,inception_v3}]
       [--pretrained]
       [--checkpoint_location CHECKPOINT_LOCATION]
       [--from_robustness]
       [--dataset DATASET]
       [--normalize_grads]
       [--save_file_location SAVE_FILE_LOCATION]
~~~

Example usages:
* Evaluation of the gradients of a standard torchvision model (without normalization):
    ~~~
    python gradient_analysis.py
           --arch resnet50
           --pretrained
           --dataset dataset/coco
           --save_file_location results/example_gradient_analysis_standard.py
    ~~~
* Evaluation of the gradients of a model trained via adversarial training from robustness (with normalization):
    ~~~
    python gradient_analysis.py
           --arch resnet50
           --checkpoint_location models/resnet50_l2_eps1.ckpt
           --from_robustness
           --dataset dataset/coco
           --normalize_grads
           --save_file_location results/example_gradient_analysis_robust.py
    ~~~