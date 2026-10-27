from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

Q_SIZE = 15
OUT_SIZE = 7
 
LAYERS = [64, 32]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        my_layers = OrderedDict()
        my_layers[f"linear_{0}"] = nn.Linear(Q_SIZE, LAYERS[0])
        for i in range(len(LAYERS)):
            layer = None
            if i == len(LAYERS) - 1:
                layer = nn.Linear(LAYERS[i], OUT_SIZE)
            else:
                layer = nn.Linear(LAYERS[i], LAYERS[i+1])
            rel = nn.ReLU()
            my_layers[f"linear_{i + 1}"] = layer
            my_layers[f"rel_{i + 1}"] = rel

        # remove last relu
        del my_layers[f"rel_{len(LAYERS)}"]
        self.linear_relu_stack = nn.Sequential(my_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_running_loss = 0.0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_running_loss += loss.detach().item()

    print(f'Loss: {train_running_loss}')
    return train_running_loss


def main():
    dataset_name = "my_dataset.npy"
    my_arr = None
    with open(dataset_name, "rb") as file_handle:
        my_arr = np.load(file_handle)

    model_path = "model.pth"
    load_model = True

    q_mes = my_arr[0]
    q_des = my_arr[1]
    qd_mes = my_arr[2]
    qd_des = my_arr[3]
    timesteps = np.arange(q_mes.shape[0]).reshape((q_mes.shape[0], 1))
    Xs = np.hstack((q_des - q_mes, qd_des - qd_mes, timesteps))
    
    tau = my_arr[6]

    # for i in range(q_mes.shape[1]):
    #     fig, axes = plt.subplots(3,1, sharex=True)
    #     axes[0].plot(timesteps, q_mes[:, i])
    #     axes[1].plot(timesteps, qd_mes[:, i])
    #     axes[2].plot(timesteps, tau[:, i])
    #     plt.show()
    tensor_x = torch.Tensor(Xs) # transform to torch tensor
    tensor_y = torch.Tensor(tau)
    my_dataset = TensorDataset(tensor_x, tensor_y)

    batch_size = 256

    # Create data loaders.
    train_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")


    model = NeuralNetwork().to(device)
    print(model)

    if load_model:
        model.load_state_dict(torch.load(model_path, weights_only=False))

    learning_rate = 1e-4
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    epochs = 100000
    save_interval = 500
    losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        losses.append(epoch_loss)
        if t % save_interval == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Saved PyTorch Model State to {model_path}")
    print("Done!")

    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")

    fig, ax = plt.subplots()
    ax.plot(np.arange(epochs), losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    fig.savefig("losses_vs_epochs.png")
    plt.show()


if __name__ == "__main__":
    main()