# Image-Classification-with-CIFAR-10

## Description

This project demonstrates image classification using the CIFAR-10 dataset with a Convolutional Neural Network (CNN) in PyTorch. The CIFAR-10 dataset contains 60,000 32x32 color images across 10 classes, including airplanes, cars, birds, and more. The goal is to build and train a CNN to classify these images effectively.

## Dataset

The CIFAR-10 dataset can be directly downloaded using torchvision, which provides a convenient way to load and preprocess the data.

## Using Google Colab

### 1. Set Up Your Google Colab Environment

- Open Google Colab and create a new notebook.
- Install necessary libraries:

    ```python
    !pip install torch torchvision
    ```

### 2. Load and Prepare the CIFAR-10 Dataset

- Import and prepare the dataset:

    ```python
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Download and load the training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Download and load the test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    ```

### 3. Define the Model

- Create a Convolutional Neural Network (CNN) model:

    ```python
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    ```

### 4. Train the Model

- Set up the training process:

    ```python
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    ```

### 5. Evaluate the Model

- Evaluate the model on the test dataset:

    ```python
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total}%')
    ```

### 6. Save and Share Your Work

- Save your trained model:

    ```python
    torch.save(net.state_dict(), 'cifar10_model.pth')
    ```

- Download the model file:

    ```python
    from google.colab import files
    files.download('cifar10_model.pth')
    ```

## Contributing

Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request.

## Contact
mbayandjambealidor@gmail.com

