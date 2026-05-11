import torch

from pydantic import BaseModel, Field
from src.utils import get_project_folder, load_config
from src.model import AbusiveHostingUseModel
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

PROJECT_ROOT = get_project_folder()


class AbusiveHostingUseInput(BaseModel):
    emails_sent_hour: float = Field(
        ge=0, description="Emails sent by hour must be greatter or equal than 0."
    )
    cpu_usage: float = Field(
        ge=0, description="CPU usage must be greatter or equal than 0."
    )
    outbound_traffic_gb: float = Field(
        ge=0, description="Outbound traffic must be greatter or equal than 0."
    )
    domains_count: float = Field(
        ge=0, description="Domains count must be greatter or equal than 0."
    )
    abuse_reports: int = Field(
        ge=0, description="Abuse reports must be greatter or equal than 0."
    )
    failed_logins_hour: float = Field(
        ge=0, description="Failed logins must be greatter or equal than 0."
    )
    requests_per_minute: float = Field(
        ge=0, description="Requests per minute must be greatter or equal than 0."
    )
    uptime_days: float = Field(
        ge=0, description="Uptime days must be greatter or equal than 0."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    parametros = load_config("config.yaml")

    MODEL_PATH = PROJECT_ROOT / parametros["model_path"]

    # Rebuild architecture
    model = AbusiveHostingUseModel(
        input_dim=int(parametros["input_dim"]), hidden_dim=int(parametros["hidden_dim"])
    )

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    app.state.model = model

    yield  # API ready


app = FastAPI(title="Abusive hosting use system.", lifespan=lifespan)


@app.post("/evaluate")
def predict(datos: AbusiveHostingUseInput):
    # Convert input to tensor
    # Pydantic already check if data has correct format
    X = torch.tensor(
        [
            [
                datos.emails_sent_hour,
                datos.cpu_usage,
                datos.outbound_traffic_gb,
                datos.domains_count,
                datos.abuse_reports,
                datos.failed_logins_hour,
                datos.requests_per_minute,
                datos.uptime_days,
            ]
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        output = app.state.model(X)
        probability = output.item()
        prediction = int(probability > 0.5)

    return {"abuse_probability": probability, "abuse_prediction": prediction}
