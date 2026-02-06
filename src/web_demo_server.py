import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Load environment variables from .env at repo root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "gesture-recognition-jemzp/5")
ROBOFLOW_INFER_URL = os.getenv(
    "ROBOFLOW_INFER_URL",
    f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}",
)

if not ROBOFLOW_API_KEY:
    # Keep server startable; endpoint will error with helpful message.
    pass

app = FastAPI(title="six-seven web demo")

# For local dev convenience. Tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "..", "public", "web-demo.html")
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(image: UploadFile = File(...)) -> dict[str, Any]:
    if not ROBOFLOW_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ROBOFLOW_API_KEY is not set. Add it to a .env file at repo root.",
        )

    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    image_bytes = await image.read()
    if len(image_bytes) > 2_500_000:
        raise HTTPException(status_code=413, detail="Image too large (max ~2.5MB)")

    params = {
        "api_key": ROBOFLOW_API_KEY,
        "format": "json",
        "confidence": os.getenv("ROBOFLOW_CONFIDENCE", "0.4"),
        "overlap": os.getenv("ROBOFLOW_OVERLAP", "0.3"),
    }

    started = time.monotonic()
    try:
        resp = requests.post(
            ROBOFLOW_INFER_URL,
            params=params,
            data=image_bytes,
            headers={"Content-Type": image.content_type},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Roboflow request failed: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Roboflow returned {resp.status_code}: {resp.text}"
        )

    payload = resp.json()
    payload["_meta"] = {
        "model_id": ROBOFLOW_MODEL_ID,
        "latency_ms": int((time.monotonic() - started) * 1000),
    }
    return payload
