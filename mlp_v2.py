# adapted from https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md
#
#

import os
import torch
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the batch size
batch_size = 4

# Transformations for the input data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.layers(x)

# Set fixed random number seed
torch.manual_seed(42)

# Prepare CIFAR-10 dataset and split into training and validation sets
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# Data loaders for training and validation sets
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
validationloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize the MLP
mlp = MLP()

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
# Adjusting learning rate from 1e-4 to 5e-4
optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4)

# Run the training loop with validation
for epoch in range(12):  # 12 epochs -> 25 epochs (Reverted to 12)
    print(f'Starting epoch {epoch+1}')
    mlp.train()  # Set the model to training mode
    current_loss = 0.0

    # Training loop
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0

    # Validation step
    mlp.eval()  # Set the model to evaluation mode
    validation_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validationloader:
            inputs, targets = data
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Print validation results
    print(f'Validation loss after epoch {epoch+1}: {validation_loss / len(validationloader)}')
    print(f'Validation accuracy: {100 * correct / total} %')

# Process is complete.
print('Training process has finished.')

# Continue with the existing code for saving, testing, etc.
# ...


## Save and Testing

PATH = './cifar_mlp_full.pth'
torch.save(mlp.state_dict(), PATH)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       transform=transform,
                                       download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


dataiter = iter(testloader)
images, labels = next(dataiter)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

outputs = mlp(images)


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = mlp(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


###
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = mlp(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


