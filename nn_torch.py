import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from  NeuralNetwork import NeuralNetwork

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(f"numero de batches {num_batches}")

    # se activa el modo de entrenamiento
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        # envia la informacion a la gpu si esque se tiene habilitada
        # si no conserva el tensor en el cpu
        X, y = X.to(device), y.to(device)

        # Calcula la prediccion del modelo
        pred = model(X)

        # Calcula el Error respecto de la funcion de perdida
        loss = loss_fn(pred, y)

        # configura a zero de los gradientes de todos los tensores optimizados
        # por el bacth anterios
        optimizer.zero_grad()

        # realiza la propagación hacia atrás.de la deribada de la perdidad
        loss.backward()

        # ajusta los valores de los pesos(w) y b
        optimizer.step()

        # cada 100 batches imprime un reporte del la perdida camculada y 
        # cuantos registros ha analizado
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model, loss_fn):
    # se obtiene el nomero de elementos en el dataset
    size = len(dataloader.dataset)

    # obtenemos el mumero de batches
    num_batches = len(dataloader)

    # se activa el modo de test/evaluacion
    model.eval()
    
    # inicializamos la perdida y la acertividad en 0 
    test_loss, correct = 0, 0

    # se desactiva el calculo del gradiente en la fase de test 
    # para optimizarl el proceso
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # ejecutamos la evaliacion con el modelo
            pred = model(X)

            # acumulamos el error generado por la funcion de perdida
            # proporcional al que corresponde por batch
            test_loss += loss_fn(pred, y).item()

            # acumulamos el valor acumulado de la acertividad proporcional 
            # a cada batch 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # dividimos el error entre el numero de batches para estimar el error promedio
    test_loss /= num_batches

    # dividimos la acertividad entre el numero de registros en el data set 
    # para obtener la acetividad promedio
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


