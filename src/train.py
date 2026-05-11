import wandb
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from model import AbusiveHostingUseModel


def train_model(
    X_train,
    input_dim=6,
    hidden_dim=16,
    lr=0.001,
    epochs=50,
    weight_decay=0.0,
    batch_size=64,
):
    model = AbusiveHostingUseModel(input_dim, hidden_dim)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    dataset = TensorDataset(X_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for (batch,) in loader:
            optimizer.zero_grad()

            outputs = model(batch)

            loss = criterion(outputs, batch)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        wandb.log(
            {
                "epoch": epoch,
                "loss": epoch_loss,
            }
        )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    return model
