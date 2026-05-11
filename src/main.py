import argparse
from pathlib import Path

import torch
import logging
import wandb

from train import train_model
from utils import (
    load_config,
    load_data,
    scale_features,
    split_train_test,
    to_tensors,
    get_project_folder,
)
from logging_config import setup_logging


def main():
    setup_logging("debug")

    logger = logging.getLogger(__name__)  # Es singleton

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="You must to add your yaml config file.")
    arguments = parser.parse_args()

    config = load_config(
        arguments.config_file,
    )

    wandb.init(
        project="abusive_hosting_use_system",
        config=config,
        job_type="training",
    )

    logger.info("Init execution.")

    # Load data
    X = load_data(config["data_path"])

    # Split
    X_train, X_test = split_train_test(
        X,
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
    )

    # Preprocess (scale values)
    x_train_s, x_test_s, scaler = scale_features(X_train, X_test)

    # Save scaler
    Path("models").mkdir(exist_ok=True)
    torch.save(scaler, config["scaler_path"])

    # Tensors
    x_train_t, x_test_t = to_tensors(x_train_s, x_test_s)

    # Train
    model = train_model(
        x_train_t,
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        lr=float(config["lr"]),
        epochs=int(config["epocas"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=int(config["batch_size"]),
    )

    # Evaluate
    with torch.no_grad():
        preds = model(x_test_t)

        reconstruction_error = torch.mean((x_test_t - preds) ** 2, dim=1)

        avg_error = reconstruction_error.mean().item()

    print(f"Average reconstruction error: {avg_error:.4f}")

    # Save model
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), config["model_path"])

    model_artifact = wandb.Artifact(
        name="trained_model", type="model", description="Trained model"
    )

    model_file = get_project_folder() / config["model_path"]
    model_artifact.add_file(model_file)
    wandb.log_artifact(model_artifact)

    print(f"Model saved in {config['model_path']}")

    logger.info("==========================")


if __name__ == "__main__":
    main()
