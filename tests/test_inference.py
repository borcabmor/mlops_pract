from fastapi.testclient import TestClient
from src.api_inference import app

client = TestClient(app)


def test_predict_endpoint():
    payload = {
        "emails_sent_hour": 100,
        "cpu_usage": 20,
        "outbound_traffic_gb": 5,
        "domains_count": 2,
        "abuse_reports": 0,
        "failed_logins_hour": 1,
        "requests_per_minute": 120,
        "uptime_days": 100,
    }

    response = client.post(
        "/evaluate",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "reconstruction_error" in data
    assert "threshold" in data
    assert "abuse_prediction" in data
