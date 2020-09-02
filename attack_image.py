import os
import torch
from PIL import Image
import argparse
import torchvision
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
                'eps': 16.0/255.0,
                'step_size': 1.0/255.0,
                'iterations': 500,
                'do_tqdm': True,
            }

        # Set the dataset for the robustness model
        dataset = ImageNet('dataset/')

        # Make a robustness model
        model, _ = make_and_restore_model(arch='resnet50', dataset=dataset,
                                          pytorch_pretrained=True)
        model = model.cuda()
        eval_model = torchvision.models.resnet50(pretrained=True).cpu().eval()

        # Get the model prediction for the original image
        label = eval_model(image.cpu())
        label = torch.argmax(label[0])
        label = label.view(1).cuda()

        # Create an adversarial example of the original images
        _, adversarial_example = model(image, label, make_adv=True, **kwargs)

        # Get the prediction of the model for the adversarial image
        adversarial_prediction = eval_model(adversarial_example.cpu())
        adversarial_prediction = torch.argmax(adversarial_prediction[0])

        # Print the original and the adversarial prediction
        print('Original prediction: ' + str(label.item()))
        print('Adversarial prediction: ' + str(adversarial_prediction.item()))

        # Save the adversarial example in the same folder as the original image
        filename_and_extension = args.image.split('.')
        adversarial_location = filename_and_extension[0] + '_adversarial.' + filename_and_extension[-1]
        save_image(adversarial_example[0], adversarial_location)

    else:
        print('Incorrect image path!')


if __name__ == '__main__':
    main()
