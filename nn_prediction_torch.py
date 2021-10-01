import torch
from torch import nn
from NeuralNetwork import NeuralNetwork
from torchvision import datasets
from torchvision.transforms import ToTensor

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]

print( test_data[0][1] )

with torch.no_grad():
    pred = model(x)

    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')