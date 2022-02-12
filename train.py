import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tqdm
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from dataset import prepare_data


def get_params():
    parser = argparse.ArgumentParser(description='FSL example code')
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", type=bool, default=True)

    params = parser.parse_args()
    params.device = torch.device("cuda" if params.use_gpu else "cpu")
    return params


def initialize_model(params, pretrained=True):
    net = models.resnet18(pretrained=pretrained)

    num_fc_in = net.fc.in_features
    net.fc = nn.Linear(num_fc_in, 2)

    net = net.to(params.device)
    return net


def train():
    params = get_params()

    trainloader, testloader = prepare_data(params)
    net = initialize_model(params, pretrained=params.pretrained)

    criterion = nn.CrossEntropyLoss()
    fc_params = list(map(id, net.fc.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, net.parameters())

    optimizer = optim.Adam([
            {'params': base_params},
            {'params': net.fc.parameters(), 'lr': params.lr * 10}],
            lr=params.lr, betas=(0.9, 0.999))

    for epoch in range(params.epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        with tqdm.tqdm(total=len(trainloader)) as pbar:
            for i, data in enumerate(trainloader):
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(params.device)
                labels = labels.to(params.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)

            eval(params, testloader, net, pbar)

    print('Finished Training')


def eval(params, testloader, net, pbar):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(params.device)
            labels = labels.to(params.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    pbar.write('\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    train()

