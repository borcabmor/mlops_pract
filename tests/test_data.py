import pandas as pd
import pytest

from src.utils import load_config, load_data

config = load_config("config.yaml")

EXPECTED_COLUMNS = [
    "emails_sent_hour",
    "cpu_usage",
    "outbound_traffic_gb",
    "domains_count",
    "abuse_reports",
    "failed_logins_hour",
    "requests_per_minute",
    "uptime_days",
]


@pytest.fixture(scope="session")
def dataframe():
    df = load_data(config["data_path"])

    return df


def test_dataset_not_empty(dataframe):
    assert len(dataframe) > 0


def test_expected_columns(dataframe):
    assert list(dataframe.columns) == EXPECTED_COLUMNS


def test_no_nulls(dataframe):
    assert dataframe.isnull().sum().sum() == 0


def test_all_columns_numeric(dataframe):
    for column in EXPECTED_COLUMNS:
        assert pd.api.types.is_numeric_dtype(dataframe[column])


def test_emails_sent_hour_positive(dataframe):
    assert (dataframe["emails_sent_hour"] >= 0).all()


def test_cpu_usage_range(dataframe):
    assert (dataframe["cpu_usage"] >= 0).all()
    assert (dataframe["cpu_usage"] <= 100).all()


def test_outbound_traffic_positive(dataframe):
    assert (dataframe["outbound_traffic_gb"] >= 0).all()


def test_domains_count_positive(dataframe):
    assert (dataframe["domains_count"] >= 0).all()


def test_failed_logins_positive(dataframe):
    assert (dataframe["failed_logins_hour"] >= 0).all()


def test_requests_per_minute_positive(dataframe):
    assert (dataframe["requests_per_minute"] >= 0).all()


def test_uptime_days_positive(dataframe):
    assert (dataframe["uptime_days"] >= 0).all()
