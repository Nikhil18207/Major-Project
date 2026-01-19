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
    """
    Enhanced in-memory rate limiter with endpoint-specific limits.

    Provides stricter limits for computationally expensive endpoints
    (report generation) and more lenient limits for lightweight endpoints.
    """

    def __init__(
        self,
        default_requests_per_minute: int = 60,
        default_requests_per_hour: int = 1000,
        # Stricter limits for expensive GPU operations
        generate_requests_per_minute: int = 10,
        generate_requests_per_hour: int = 100,
    ):
        # Default limits for most endpoints
        self.default_rpm = default_requests_per_minute
        self.default_rph = default_requests_per_hour

        # Stricter limits for generation endpoints (GPU-intensive)
        self.generate_rpm = generate_requests_per_minute
        self.generate_rph = generate_requests_per_hour

        # Separate tracking for different endpoint types
        self.default_minute_requests = defaultdict(list)
        self.default_hour_requests = defaultdict(list)
        self.generate_minute_requests = defaultdict(list)
        self.generate_hour_requests = defaultdict(list)

        self._lock = threading.Lock()

        # Endpoints that are computationally expensive
        self.expensive_endpoints = {
            "/api/v1/generate",
            "/api/v1/generate/base64",
            "/api/v1/generate/batch",
            "/api/v1/generate/analyze",
            "/api/v1/attention/visualize",
        }

    def _cleanup_old_requests(self, requests_dict: dict, client_id: str, max_age: float, current_time: float):
        """Remove expired request timestamps."""
        cutoff = current_time - max_age
        requests_dict[client_id] = [t for t in requests_dict[client_id] if t > cutoff]

    def is_allowed(self, client_id: str, path: str = "") -> tuple:
        """
        Check if request is allowed for the client.

        Args:
            client_id: Unique client identifier (usually IP)
            path: Request path to determine rate limit tier

        Returns:
            Tuple of (is_allowed, error_message, retry_after_seconds)
        """
        with self._lock:
            current_time = time.time()

            # Determine if this is an expensive endpoint
            is_expensive = any(path.startswith(ep) for ep in self.expensive_endpoints)

            if is_expensive:
                # Strict limits for GPU-intensive operations
                minute_requests = self.generate_minute_requests
                hour_requests = self.generate_hour_requests
                rpm_limit = self.generate_rpm
                rph_limit = self.generate_rph
                limit_type = "generation"
            else:
                # Default limits for other endpoints
                minute_requests = self.default_minute_requests
                hour_requests = self.default_hour_requests
                rpm_limit = self.default_rpm
                rph_limit = self.default_rph
                limit_type = "default"

            # Cleanup old requests
            self._cleanup_old_requests(minute_requests, client_id, 60, current_time)
            self._cleanup_old_requests(hour_requests, client_id, 3600, current_time)

            # Check minute limit
            if len(minute_requests[client_id]) >= rpm_limit:
                retry_after = 60 - (current_time - minute_requests[client_id][0])
                return False, f"Rate limit exceeded ({limit_type}): {rpm_limit} requests/minute", int(retry_after) + 1

            # Check hour limit
            if len(hour_requests[client_id]) >= rph_limit:
                retry_after = 3600 - (current_time - hour_requests[client_id][0])
                return False, f"Rate limit exceeded ({limit_type}): {rph_limit} requests/hour", int(retry_after) + 1

            # Record request
            minute_requests[client_id].append(current_time)
            hour_requests[client_id].append(current_time)

            return True, None, 0

    def get_remaining(self, client_id: str, path: str = "") -> dict:
        """Get remaining requests for a client."""
        with self._lock:
            current_time = time.time()
            is_expensive = any(path.startswith(ep) for ep in self.expensive_endpoints)

            if is_expensive:
                minute_requests = self.generate_minute_requests
                hour_requests = self.generate_hour_requests
                rpm_limit = self.generate_rpm
                rph_limit = self.generate_rph
            else:
                minute_requests = self.default_minute_requests
                hour_requests = self.default_hour_requests
                rpm_limit = self.default_rpm
                rph_limit = self.default_rph

            # Cleanup and count
            self._cleanup_old_requests(minute_requests, client_id, 60, current_time)
            self._cleanup_old_requests(hour_requests, client_id, 3600, current_time)

            return {
                "remaining_per_minute": max(0, rpm_limit - len(minute_requests[client_id])),
                "remaining_per_hour": max(0, rph_limit - len(hour_requests[client_id])),
                "limit_per_minute": rpm_limit,
                "limit_per_hour": rph_limit,
            }


# Global rate limiter instance with configurable limits via environment
rate_limiter = RateLimiter(
    default_requests_per_minute=int(os.environ.get("RATE_LIMIT_DEFAULT_RPM", "60")),
    default_requests_per_hour=int(os.environ.get("RATE_LIMIT_DEFAULT_RPH", "1000")),
    generate_requests_per_minute=int(os.environ.get("RATE_LIMIT_GENERATE_RPM", "10")),
    generate_requests_per_hour=int(os.environ.get("RATE_LIMIT_GENERATE_RPH", "100")),
)


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
    """Apply rate limiting to API endpoints with endpoint-specific limits."""
    # Skip rate limiting for health checks and docs
    skip_paths = ["/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"]
    if request.url.path in skip_paths:
        return await call_next(request)

    # Get client identifier (IP address or forwarded IP)
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    # Check rate limit with path for endpoint-specific limits
    is_allowed, error_message, retry_after = rate_limiter.is_allowed(client_ip, request.url.path)
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for {client_ip} on {request.url.path}: {error_message}")
        from fastapi.responses import JSONResponse

        # Get remaining limits for headers
        remaining = rate_limiter.get_remaining(client_ip, request.url.path)

        return JSONResponse(
            status_code=429,
            content={
                "detail": error_message,
                "retry_after": retry_after,
                "limits": remaining,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(remaining["limit_per_minute"]),
                "X-RateLimit-Remaining": str(remaining["remaining_per_minute"]),
            }
        )

    # Add rate limit headers to successful responses
    response = await call_next(request)
    remaining = rate_limiter.get_remaining(client_ip, request.url.path)
    response.headers["X-RateLimit-Limit"] = str(remaining["limit_per_minute"])
    response.headers["X-RateLimit-Remaining"] = str(remaining["remaining_per_minute"])

    return response


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
