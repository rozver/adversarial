import torch
import torchvision


def get_original_pred_score(model, x, true_label):
    output = model(x.cuda())
    output = torch.nn.Softmax()(output)
    prediction = output[0, true_label]
    return prediction


def attack_pixels(model, x, true_label, num_iters=1000, epsilon=0.6):
    n_dims = x.view(-1).size(0)
    perm = torch.randperm(n_dims)
    last_prob = get_original_pred_score(model, x, true_label)
    for i in range(num_iters):
        diff = torch.zeros(n_dims)
        diff[perm[i]] = epsilon
        left_prob = get_original_pred_score(model, (x - diff.view(x.size())).clamp(0, 1), true_label)
        if left_prob < last_prob:
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob = left_prob
        else:
            right_prob = get_original_pred_score(model, (x + diff.view(x.size())).clamp(0, 1), true_label)
            if right_prob < last_prob:
                x = (x + diff.view(x.size())).clamp(0, 1)
                last_prob = right_prob
    return x


def get_probs(model, x, y):
    output = model(x.cuda()).cpu()
    probs = torch.nn.Softmax()(output)[:, y]
    return torch.diag(probs.data)


def simba_single(model, x, y, num_iters=300, epsilon=0.2):
    n_dims = x.view(1, -1).size(1)
    perm = torch.randperm(n_dims)
    last_prob = get_probs(model, x, y)
    for i in range(num_iters):
        diff = torch.zeros(n_dims)
        diff[perm[i]] = epsilon
        left_prob = get_probs(model, (x - diff.view(x.size())).clamp(0, 1), y)
        if left_prob < last_prob:
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob = left_prob
        else:
            right_prob = get_probs(model, (x + diff.view(x.size())).clamp(0, 1), y)
            if right_prob < last_prob:
                x = (x + diff.view(x.size())).clamp(0, 1)
                last_prob = right_prob
    return x


def main():
    pixels_score = 0
    simba_score = 0
    model = torchvision.models.resnet50(pretrained=True).cuda().eval()
    data_loader = torch.load('dataset/imagenet-dogs-images.pt')
    for images_batch in data_loader:
        for image in images_batch:
            original_prediction = torch.argmax(model(image.view(1, 3, 224, 224).cuda())).item()
            adversarial_example_attack_pixels = attack_pixels(model, image.view(1, 3, 224, 224), original_prediction)
            adversarial_prediction_pixels = torch.argmax(
                model(adversarial_example_attack_pixels.view(1, 3, 224, 224).cuda())
            ).item()

            adversarial_example_simba = simba_single(model, image.view(1, 3, 224, 224), 1)
            adversarial_prediction_simba = torch.argmax(
                model(adversarial_example_simba[0].view(1, 3, 224, 224).cuda())
            ).item()

            if original_prediction != adversarial_prediction_pixels:
                pixels_score += 1
            if original_prediction != adversarial_prediction_simba:
                simba_score += 1

    print('Pixels score: ' + str(pixels_score))
    print('Simba score: ' + str(simba_score))


if __name__ == '__main__':
    main()
