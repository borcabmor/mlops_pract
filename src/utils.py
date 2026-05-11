from pathlib import Path

import pandas as pd
import torch
import yaml
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    logger = logging.getLogger(__name__)

    logger.info("Loading dataset.")

    return pd.read_csv(path)


def split_train_test(X, test_size=0.2, random_state=42):
    return train_test_split(X, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    return x_train_scaled, x_test_scaled, scaler


def to_tensors(X_train, X_test):
    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    x_test_t = torch.tensor(X_test, dtype=torch.float32)

    return (
        x_train_t,
        x_test_t,
    )


def get_project_folder() -> Path:
    return Path(__file__).resolve().parents[1]


def get_config_file_path():
    raiz_project = get_project_folder()

    return raiz_project / "config/"


def load_config(filename: str) -> dict:
    logger = logging.getLogger(__name__)

    logger.warning("Loading configuration.")

    raiz_project = get_project_folder()
    config_file = raiz_project / "config/" / filename

    with open(config_file, "r") as fichero:
        return yaml.safe_load(fichero)
