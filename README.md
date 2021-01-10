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
Example usage - creating robust adversarial examples (utilizing [EOT](https://arxiv.org/abs/1707.07397))
of images from COCO dataset ('airplane' category)against ResNet50 by using our proposed foreground-only attack:
~~~
python pgd.py
       --arch resnet50
       --dataset dataset/coco/airplane.pt
       --masks
       --eps 1.0
       --norm l2
       --step_size 1/255.0
       --num_iterations 50
       --transfer
       --eot
       --save_file_location results/example_pgd.pt
~~~

#### Model training (adversarial training featured)

Example usage - adversarial training from scratch on part of the ImageNet dataset (airplanes):
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
* Noise addition
* Rotation
* Translation

Example usage - rotating the image 'dog.png':

~~~
python transformations.py --image dog.png --transformation_type rotation
~~~

### Blackbox attacks
Example usages of different attacks implemented by us:

* Transferable attack (foreground-only version):
    ~~~
    python pgd.py
       --arch resnet50
       --dataset dataset/coco/airplane.pt
       --masks
       --eps 1.0
       --norm l2
       --step_size 1/255.0
       --num_iterations 50
       --transfer
       --save_file_location results/example_pgd.pt
    ~~~

* [FGSM](https://arxiv.org/abs/1412.6572) [NES](https://arxiv.org/abs/1106.4487):
    ~~~
    python blackbox.py 
           --arch resnet50
           --dataset dataset/imagenet-airplanes-images.pt
           --attack_type nes
           --eps 4
           --num_iterations 1000
           --save_file_location results/example_nes.pt
    ~~~
* [SimBA](https://arxiv.org/abs/1905.07121) (original):
    ~~~
    python blackbox.py 
           --arch resnet50
           --dataset dataset/imagenet-airplanes-images.pt
           --attack_type simba
           --eps 4
           --num_iterations 1000
           --save_file_location results/example_simba.pt
    ~~~

* Gradient-based SimBA (ours):
    ~~~
    python blackbox.py 
           --model resnet50
           --dataset dataset/imagenet-airplanes-images.pt
           --gradient_masks
           --grqdient-model resnet18
           --attack_type simba
           --eps 4
           --num_iterations 1000
           --save_file_location results/example_simba.pt
    ~~~
  
