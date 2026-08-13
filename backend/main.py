from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from backend.inference import (
    inference_engine,
    CLASS_NAMES,
)


app = FastAPI(
    title="HELIO-FORGE AI API",
    description="Solar Flare Intelligence API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():

    return {
        "name": "HELIO-FORGE AI",
        "status": "online",
        "service": "Solar Flare Intelligence API",
    }
class PredictionRequest(BaseModel):
    sequence: list[list[float]]


@app.post("/api/predict")
def predict(request: PredictionRequest):

    tensor = torch.tensor(
        [request.sequence],
        dtype=torch.float32,
    )

    return inference_engine.predict_tensor(tensor)

@app.get("/api/health")
def health():

    checkpoint = inference_engine.checkpoint
    args = checkpoint["args"]

    val_metrics = checkpoint.get(
        "val_metrics",
        {},
    )

    return {
        "status": "ok",
        "model": "HelioForgeTCN",
        "checkpoint": "best_macro_f1.pt",
        "epoch": checkpoint.get("epoch"),
        "macro_f1": val_metrics.get(
            "macro_f1"
        ),
        "input_shape": [
            args["in_channels"],
            512,
        ],
        "classes": CLASS_NAMES,
    }


@app.get("/api/demo")
def demo():

    return inference_engine.predict_demo(0)


@app.get("/api/demo/{index}")
def demo_index(index: int):

    return inference_engine.predict_demo(index)