# app.py - FastAPI example with Prometheus metrics
from fastapi import FastAPI, Request, Response
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram, Gauge
import json
import time
from math import exp
import uvicorn
import random
from typing import Dict, Any, List
from contextlib import asynccontextmanager

# Define custom metrics for ML-specific instrumentation
ML_TRAINING_COUNTER = Counter(
    name="ml_training_count_total",
    documentation="Number of model training operations",
    labelnames=["status"]
)

ML_PREDICTION_COUNTER = Counter(
    name="ml_prediction_count_total",
    documentation="Number of prediction requests",
    labelnames=["status"]
)

ML_PREDICTION_LATENCY = Histogram(
    name="ml_prediction_latency",
    documentation="Prediction latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

ML_ACCURACY = Gauge('ml_accuracy', 'Accuracy of the predictor')
ML_SAMPLES_PROCESSED = Gauge('ml_train_samples_processed', 'Number of samples processed')

# Define lifespan context manager for the app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Shutdown logic

# Create FastAPI app with lifespan
app = FastAPI(title="ML API with Prometheus Metrics", lifespan=lifespan)

# Setup Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Middleware to measure request duration and update metrics
@app.middleware("http")
async def add_process_time_and_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Update metrics based on request path
    if request.url.path == "/predict/":
        ML_PREDICTION_LATENCY.observe(process_time)
        ML_PREDICTION_COUNTER.labels(
            status="success" if response.status_code < 400 else "failure"
        ).inc()
            
    elif request.url.path == "/train_predictor/":
        ML_TRAINING_COUNTER.labels(
            status="success" if response.status_code < 400 else "failure"
        ).inc()

    return response

@app.post("/post_data/")
async def post_data(data: Dict[str, Any]):
    time.sleep(random.uniform(0.05, 0.2))
    return {"status": "data received", "data_points": len(data)}


@app.post("/train_predictor/")
async def train_predictor(training_data: List[Dict[str, Any]]):
    training_time = exp(random.uniform(-3, 3.0))
    time.sleep(training_time)
    accuracy = random.uniform(0.7, 0.99)
    samples_processed = len(training_data)
    
    ML_ACCURACY.set(accuracy)
    ML_SAMPLES_PROCESSED.set(samples_processed)

    return {
        "status": "model trained", 
        "training_time": training_time,
        "accuracy": accuracy,
        "samples_processed": samples_processed,
    }

@app.post("/predict/")
async def predict(features: Dict[str, Any]):
    if random.random() < .1:
        raise ValueError("Prediction failed")

    inference_time = random.uniform(0.01, 0.5)
    time.sleep(inference_time)
    prediction = random.uniform(0, 1)
    confidence = random.uniform(0.6, 0.98)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "inference_time": inference_time
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
