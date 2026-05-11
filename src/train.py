from model import AbusiveHostingUseModel
import torch.nn as nn
import torch.optim as optim
import wandb


def train_model(
    X_train, input_dim=6, hidden_dim=16, lr=0.001, epochs=50, weight_decay=0.0
):
    model = AbusiveHostingUseModel(input_dim, hidden_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, X_train)
        loss.backward()
        optimizer.step()

        wandb.log({"epoch": epoch, "loss": loss.item()})

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
