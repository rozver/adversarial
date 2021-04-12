# Black-box adversarial examples in the real world

Code for the paper "Black-box adversarial examples in the real world", authored by 
[Hristo Todorov](https://github.com/RoZvEr/) under the supervision of 
[Kristian Georgiev](https://github.com/kristian-georgiev/).

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
#### Projected Gradient Descent
* Example usage: creating adversarial examples against ResNet50, which are constrained by &epsilon; = 1 (l<sub>2</sub> norm):
~~~
python pgd.py
       --arch resnet50
       --dataset dataset/imagenet
       --eps 1.0
       --norm l2
       --step_size 1/255.0
       --num_iterations 50
       --save_file_location results/example_pgd.pt
~~~

### Foreground Attack (ours) 
* Example usage: creating robust adversarial examples (utilizing [EOT](https://arxiv.org/abs/1707.07397)) against ResNet50:
~~~
python pgd.py
       --arch resnet50
       --dataset dataset/coco/airplane.pt
       --masks
       --eps 8
       --norm linf
       --step_size 1/255.0
       --num_iterations 50
       --save_file_location results/example_pgd.pt
~~~


#### Model training (adversarial training featured)

Example usage - adversarial training from scratch on part of the ImageNet dataset (airplanes):
~~~
python train.py
       --arch resnet50
       --dataset dataset/imagenet
       --epochs 50
       --learning_rate 0.01
       --adversarial
       --save_file_location models/example_robust_model.pt
~~~

### Gradient analysis
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

### Transformations
Our transformation framework supports the following transformation types:
* Light adjustment
* Gaussian noise
* Gaussian blur
* Translation
* Rotation

Example usage - rotating the image 'dog.png':

~~~
python transformations.py --image dog.png --transformation_type rotation
~~~

### Blackbox attacks
Example usages of different attacks implemented by us:
#### [FGSM](https://arxiv.org/abs/1412.6572) [NES](https://arxiv.org/abs/1106.4487) (Goodfellow et al. 2015, Wierstra et al. 2011):
    ~~~
    python blackbox.py 
           --arch resnet50
           --dataset dataset/imagenet
           --attack_type nes
           --eps 4
           --num_iterations 1000
           --save_file_location results/example_nes.pt
    ~~~
#### [SimBA](https://arxiv.org/abs/1905.07121) (Guo et al. 2019):
    ~~~
    python blackbox.py 
           --arch resnet50
           --dataset dataset/imagenet
           --attack_type simba
           --eps 4
           --num_iterations 1000
           --save_file_location results/example_simba.pt
    ~~~

  
 #### Selective Transfer Attack (ours):
 * 50 iterations of evaluation over a Gaussian distribution with &sigma; = 25/255
    ~~~
    python pgd.py
       --arch resnet50
       --dataset dataset/imagenet
       --eps 4
       --norm linf
       --step_size 1/255.0
       --num_iterations 50
       --transfer
       -- selective
       --num_transformations 25
       --sigma 25
       --save_file_location results/example_selective_transfer.pt
    ~~~


#### Gradient-based SimBA (ours)
* Single substitute model (ResNet18):
    ~~~
    python blackbox.py 
           --model resnet50
           --dataset dataset/imagenet
           --gradient_priors
           --substitute_model resnet18
           --attack_type simba
           --eps 0.06
           --num_iterations 1000
           --save_file_location results/example_simba_single.pt
    ~~~
  
* A set of substitute models (sampled using the same method, which is
 present in our Selective Transfer Attack):
    ~~~
    python blackbox.py 
           --model resnet50
           --dataset dataset/imagenet
           --gradient_priors
           --ensemble_selection
           --attack_type simba
           --eps 0.06
           --num_iterations 1000
           --save_file_location results/example_simba_ensemble.pt
    ~~~