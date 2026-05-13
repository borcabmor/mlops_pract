import torch
import pandas as pd

from src.model import AbusiveHostingUseModel
from src.utils import (
    split_train_test,
    scale_features,
    to_tensors,
)
from src.train import train_model
from unittest.mock import patch


# UTILS TESTS
def test_split_train_test():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
    )

    train, test = split_train_test(df, test_size=0.2)

    assert len(train) == 4
    assert len(test) == 1


def test_scale_features():
    train = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [10, 20, 30],
        }
    )

    test = pd.DataFrame(
        {
            "a": [4],
            "b": [40],
        }
    )

    x_train_s, x_test_s, scaler = scale_features(train, test)

    assert x_train_s.shape == (3, 2)
    assert x_test_s.shape == (1, 2)

    assert scaler is not None


def test_to_tensors():
    x_train = [[1.0, 2.0], [3.0, 4.0]]
    x_test = [[5.0, 6.0]]

    x_train_t, x_test_t = to_tensors(x_train, x_test)

    assert isinstance(x_train_t, torch.Tensor)
    assert isinstance(x_test_t, torch.Tensor)

    assert x_train_t.dtype == torch.float32


# MODEL TESTS
def test_model_forward():
    model = AbusiveHostingUseModel(
        input_dim=8,
        hidden_dim=16,
    )

    x = torch.rand(4, 8)
    output = model(x)

    assert output.shape == (4, 8)


def test_model_output_is_tensor():
    model = AbusiveHostingUseModel(
        input_dim=8,
        hidden_dim=16,
    )

    x = torch.rand(1, 8)

    output = model(x)

    assert isinstance(output, torch.Tensor)


# TRAIN TESTS
@patch("src.train.wandb.log")
def test_train_model_returns_model():
    x_train = torch.rand(100, 8)

    model = train_model(
        x_train,
        input_dim=8,
        hidden_dim=16,
        epochs=1,
        batch_size=16,
    )

    assert model is not None


@patch("src.train.wandb.log")
def test_train_model_forward_after_training():
    x_train = torch.rand(100, 8)

    model = train_model(
        x_train,
        input_dim=8,
        hidden_dim=16,
        epochs=1,
        batch_size=16,
    )

    output = model(x_train[:5])

    assert output.shape == (5, 8)
