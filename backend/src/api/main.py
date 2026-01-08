"""
XR2Text FastAPI Application

Main entry point for the REST API server that provides
endpoints for X-ray report generation and model inference.
"""

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
import time

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from ..models.xr2text import XR2TextModel, DEFAULT_CONFIG
from ..utils.device import get_device, print_gpu_info, setup_cuda_optimizations
from ..utils.logger import setup_logger
from .routes import router


# ============================================
# Rate Limiting Configuration
# ============================================
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 30, requests_per_hour: int = 500):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = defaultdict(list)
        self.hour_requests = defaultdict(list)
        self._lock = threading.Lock()

    def _cleanup_old_requests(self, client_id: str, current_time: float):
        """Remove expired request timestamps."""
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        self.minute_requests[client_id] = [
            t for t in self.minute_requests[client_id] if t > minute_ago
        ]
        self.hour_requests[client_id] = [
            t for t in self.hour_requests[client_id] if t > hour_ago
        ]

    def is_allowed(self, client_id: str) -> tuple:
        """
        Check if request is allowed for the client.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        with self._lock:
            current_time = time.time()
            self._cleanup_old_requests(client_id, current_time)

            # Check minute limit
            if len(self.minute_requests[client_id]) >= self.requests_per_minute:
                return False, f"Rate limit exceeded: {self.requests_per_minute} requests/minute"

            # Check hour limit
            if len(self.hour_requests[client_id]) >= self.requests_per_hour:
                return False, f"Rate limit exceeded: {self.requests_per_hour} requests/hour"

            # Record request
            self.minute_requests[client_id].append(current_time)
            self.hour_requests[client_id].append(current_time)

            return True, None


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500)


# Environment-based CORS configuration
def get_cors_origins() -> List[str]:
    """Get CORS origins based on environment."""
    env = os.environ.get("ENV", "development")

    # Default development origins
    dev_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    if env == "production":
        # In production, only allow specific origins
        cors_env = os.environ.get("CORS_ORIGINS", "")
        if cors_env:
            allowed_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
            if allowed_origins:
                return allowed_origins
        # Fallback: If no CORS_ORIGINS set in production, log warning and use restrictive default
        logger.warning("CORS_ORIGINS not set in production. Using restrictive defaults.")
        return dev_origins  # Return dev origins as fallback (better than ["*"])
    else:
        # In development, allow localhost origins
        return dev_origins


# ============================================
# Thread-Safe Global Model Instance
# ============================================
class ModelManager:
    """Thread-safe model manager for handling model access."""

    def __init__(self):
        self._model: Optional[XR2TextModel] = None
        self._device: Optional[torch.device] = None
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self, model: XR2TextModel, device: torch.device):
        """Initialize the model (called once at startup)."""
        with self._lock:
            self._model = model
            self._device = device
            self._initialized = True

    def get_model(self) -> XR2TextModel:
        """Get the model instance (thread-safe)."""
        with self._lock:
            if not self._initialized or self._model is None:
                raise RuntimeError("Model not loaded. Server may still be starting.")
            return self._model

    def get_device(self) -> torch.device:
        """Get the device instance (thread-safe)."""
        with self._lock:
            if self._device is None:
                self._device = get_device()
            return self._device

    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        with self._lock:
            return self._initialized and self._model is not None


# Global model manager instance
model_manager = ModelManager()

# Legacy global variables for backward compatibility
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

    # Initialize thread-safe model manager
    model_manager.initialize(model, device)

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

# CORS middleware for frontend access (environment-aware)
cors_origins = get_cors_origins()
logger.info(f"CORS origins configured: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================
# Rate Limiting Middleware
# ============================================
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to API endpoints."""
    # Skip rate limiting for health checks and docs
    if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    # Get client identifier (IP address or forwarded IP)
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    # Check rate limit
    is_allowed, error_message = rate_limiter.is_allowed(client_ip)
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for {client_ip}: {error_message}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={"detail": error_message},
            headers={"Retry-After": "60"}
        )

    return await call_next(request)


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
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "healthy",
        "model_loaded": model_manager.is_ready(),
        "device": str(model_manager.get_device()) if model_manager.is_ready() else None,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda if gpu_available else None,
    }


def get_model() -> XR2TextModel:
    """
    Get the global model instance (thread-safe).

    Uses the ModelManager for thread-safe access.
    Falls back to legacy global for backward compatibility.
    """
    # Prefer thread-safe model manager
    if model_manager.is_ready():
        return model_manager.get_model()

    # Fallback to legacy global
    global model
    if model is None:
        raise RuntimeError("Model not loaded. Server may still be starting.")
    return model


def get_device_instance() -> torch.device:
    """
    Get the global device instance (thread-safe).

    Uses the ModelManager for thread-safe access.
    """
    return model_manager.get_device()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
