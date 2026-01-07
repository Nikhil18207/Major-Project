"""
XR2Text FastAPI Application

Main entry point for the REST API server that provides
endpoints for X-ray report generation and model inference.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from ..models.xr2text import XR2TextModel, DEFAULT_CONFIG
from ..utils.device import get_device, print_gpu_info, setup_cuda_optimizations
from ..utils.logger import setup_logger
from .routes import router


# Global model instance
model: Optional[XR2TextModel] = None
device: Optional[torch.device] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.

    Loads the model on startup and cleans up on shutdown.
    """
    global model, device

    # Setup logging
    setup_logger(log_dir="logs", log_level="INFO")

    logger.info("Starting XR2Text API server...")

    # Setup device
    device = get_device()
    print_gpu_info()
    setup_cuda_optimizations()

    # Load model
    checkpoint_path = os.environ.get("MODEL_CHECKPOINT", None)

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = XR2TextModel.from_pretrained(checkpoint_path)
    else:
        logger.info("Loading model with default configuration...")
        model = XR2TextModel.from_config(DEFAULT_CONFIG)

    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully!")
    logger.info(f"Device: {device}")

    yield

    # Cleanup
    logger.info("Shutting down XR2Text API server...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="XR2Text API",
    description="End-to-End Transformer Framework for Chest X-Ray Report Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "XR2Text API",
        "version": "1.0.0",
        "description": "Chest X-Ray Report Generation API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model, device

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda if gpu_available else None,
    }


def get_model() -> XR2TextModel:
    """Get the global model instance."""
    global model
    if model is None:
        raise RuntimeError("Model not loaded")
    return model


def get_device_instance() -> torch.device:
    """Get the global device instance."""
    global device
    if device is None:
        device = get_device()
    return device


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
