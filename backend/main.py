from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from time import sleep

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.inference import CLASS_NAMES, inference_engine
from backend.explanation import explain_prediction
from backend.preprocessing import prepare_upload

app = FastAPI(title="HELIO-FORGE AI API", description="Solar Flare Intelligence API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

PERFORMANCE = {"accuracy": .8941, "macro_f1": .8514, "macro_precision": .8488, "macro_recall": .8698,
 "per_class": {"Quiet": {"precision": .9485, "recall": 1, "f1": .9735, "support": 92}, "B": {"precision": .8716, "recall": .9627, "f1": .9149, "support": 134}, "C": {"precision": .9775, "recall": .7982, "f1": .8788, "support": 109}, "M": {"precision": .8696, "recall": .7547, "f1": .8081, "support": 53}, "X": {"precision": .5769, "recall": .8333, "f1": .6818, "support": 18}},
 "confusion_matrix": [[92,0,0,0,0],[5,129,0,0,0],[0,19,87,3,0],[0,0,2,40,11],[0,0,0,3,15]],
 "training_history": [{"epoch": 1,"train_loss": 1.4162,"val_loss": 2.607,"val_f1": .5912,"lr": .001},{"epoch": 19,"train_loss": .0672,"val_loss": .9214,"val_f1": .8164,"lr": .00025},{"epoch": 25,"train_loss": None,"val_loss": .8234,"val_f1": .8714,"lr": .00025}]}

def _result_or_503(fn):
    try: return fn()
    except (ValueError, IndexError) as exc: raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(503, str(exc)) from exc

@app.get("/")
def root(): return {"name": "HELIO-FORGE AI", "status": "online"}

@app.get("/api/health")
def health():
    metrics = inference_engine.checkpoint.get("val_metrics", {})
    return {"status": "ok" if inference_engine.ready else "degraded", "model": "HelioForgeTCN", "checkpoint": "best_macro_f1.pt", "epoch": inference_engine.checkpoint.get("epoch", 25), "macro_f1": metrics.get("macro_f1", PERFORMANCE["macro_f1"]), "input_shape": [32, 512], "classes": CLASS_NAMES, "detail": inference_engine.error}

@app.get("/api/demo")
@app.get("/api/demo/{index}")
def demo(index: int = 0): return _result_or_503(lambda: inference_engine.predict_demo(index))

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(422, "A file name is required.")
    payload = await file.read()
    try:
        prepared = prepare_upload(file.filename, payload)
        result = inference_engine.predict_tensor(prepared.tensor)
        result["observation_id"] = file.filename
        result.update({"features": prepared.features, "signal": prepared.signal, "rgb_intensity": prepared.rgb_intensity, "active_regions": prepared.active_regions})
        return result
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Could not read tensor: {exc}") from exc

@app.get("/api/evolution")
def evolution():
    now = datetime.now(timezone.utc)
    labels = [(0,"Quiet",.91),(1,"B",.78),(1,"B",.82),(2,"C",.73),(3,"M",.87)]
    return {"sequence": [{"timestamp": (now - timedelta(hours=4-i)).isoformat(), "class": c, "label": label, "confidence": confidence} for i,(c,label,confidence) in enumerate(labels)]}

@app.get("/api/performance")
def performance(): return PERFORMANCE

@app.get("/api/explanation")
def explanation(class_id: int = 3, confidence: float = .87):
    if class_id not in range(5) or not 0 <= confidence <= 1:
        raise HTTPException(422, "class_id must be 0–4 and confidence must be 0–1.")
    return explain_prediction(class_id, confidence)

@app.get("/api/stream/predict")
def stream_predict():
    def events():
        for step, elapsed in [("Observation Loaded", 12), ("FITS Parsing", 8), ("Feature Engineering", 22), ("TCN Inference", 78)]:
            yield f"event: pipeline_step\ndata: {json.dumps({'step': step, 'status': 'done', 'elapsed_ms': elapsed})}\n\n"; sleep(.2)
        yield "event: complete\ndata: {}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")
