import os
import torch
from PIL import Image
import argparse
from torchvision import transforms
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
from torchvision.utils import save_image


def main():
    # Parse image location from command argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    # Check if the given location exists and it is a valid image file
    if os.path.exists(args.image) and args.image.endswith(('png', 'jpg', 'jpeg')):
        # Open the image from the given location
        image = Image.open(args.image)

        # Transform the image to a PyTorch tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = transform(image)
        image = image.unsqueeze(0).cuda()

        # Attack parameters
        kwargs = {
                'constraint': 'inf',
                'eps': 16.0/255,
                'step_size': 1/255.0,
                'iterations': 500,
                'do_tqdm': True,
                'targeted': False,
            }

        # Set the dataset for the robustness model
        dataset = ImageNet('dataset/imagenet-airplanes')

        # Make a robustness model
        model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                          pytorch_pretrained=True)
        model = model.eval().cuda()

        # Get the model prediction for the original image
        label = model(image)
        label = torch.argmax(label[0])
        label = label.view(1).cuda()

        # Create an adversarial example of the original images
        _, images_adversarial = model(image, label, make_adv=True, **kwargs)
        predictions, _ = model(images_adversarial)
        print(torch.argmax(predictions[0]))

        # Save the adversarial example in the same folder as the original iamge
        adversarial_location = args.image[0:-5] + '-adversarial' + args.image[-5:]
        save_image(images_adversarial[0], adversarial_location)

    else:
        print('Incorrect image path!')


if __name__ == '__main__':
    main()
