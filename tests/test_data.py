import pytest

from src.utils import load_config, load_data

config = load_config("config.yaml")


@pytest.fixture(scope="session")
def dataframe():
    df = load_data(config["data_path"])

    return df


def test_dataset_not_empty(dataframe):
    assert len(dataframe) > 0
