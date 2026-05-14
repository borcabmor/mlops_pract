# Abusive Hosting Use Detection

Proyecto de MLOps para detección de uso abusivo de servicios de hosting mediante un modelo Autoencoder desarrollado con PyTorch.

---

## Funcionalidades principales

- Entrenamiento de un modelo de Machine Learning con PyTorch.
- API REST desarrollada con FastAPI.
- Detección de anomalías basada en error de reconstrucción.
- Contenerización mediante Docker.
- Tests automáticos con Pytest.
- Pipeline CI/CD con GitHub Actions.
- Despliegue automático en Render.
- Monitorización de experimentos con Weights & Biases.

---

## Estructura del proyecto

```text
.
├── src/
├── tests/
├── models/
├── config/
├── Dockerfile.api
├── requirements.txt
└── README.md
```

---

## Instalación del entorno local

Clonar el repositorio:

```bash
git clone https://github.com/borcabmor/mlops_pract.git
cd mlops_pract
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Entrenamiento del modelo

```bash
python -m src.main config.yaml
```

---

## Weights & Biases

https://wandb.ai/borcabmor-taya/abusive_hosting_use_system

---

## Lanzar API localmente

```bash
uvicorn src.api_inference:app --host 0.0.0.0 --port 8000
```

La API estará disponible en:

```text
http://localhost:8000
```

Documentación Swagger:

```text
http://localhost:8000/docs
```

---

## Docker

Construir imagen:

```bash
docker build -f Dockerfile.api -t abusive-hosting-api .
```

Ejecutar contenedor:

```bash
docker run -p 8000:8000 abusive-hosting-api
```

---

## Tests

```bash
pytest tests/
```

---

## Endpoint accesible en producción (Render)

El despliegue en Render se realiza automáticamente desde GitHub Actions.

Endpoint:

```text
https://abusive-hosting-api-latest.onrender.com/evaluate
```

Ejemplo de petición POST:

```json
{
    "emails_sent_hour": 129.345,
    "cpu_usage": 19.966,
    "outbound_traffic_gb": 9.345,
    "domains_count": 2.646,
    "abuse_reports": 0,
    "failed_logins_hour": 3.345,
    "requests_per_minute": 120.345,
    "uptime_days": 200.345
}
```

---

## Autor

- Borja Cabañas Morales