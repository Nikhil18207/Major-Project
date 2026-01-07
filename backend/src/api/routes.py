"""
API Routes for XR2Text with HAQT-ARR

Defines all API endpoints for report generation, model info,
attention visualization, and feedback collection.

Features:
- Report generation from X-ray images
- HAQT-ARR attention visualization (Novel)
- Anatomical region importance analysis
- Batch processing support

Authors: S. Nikhil, Dadhania Omkumar
Supervisor: Dr. Damodar Panigrahy
"""

import io
import base64
import time
from typing import Optional, List
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from ..data.transforms import get_val_transforms, XRayTransform


router = APIRouter()


# ============================================
# Pydantic Models for Request/Response
# ============================================

class ReportGenerationRequest(BaseModel):
    """Request model for report generation from base64 image."""
    image_base64: str = Field(..., description="Base64 encoded X-ray image")
    max_length: int = Field(256, ge=50, le=512, description="Maximum report length")
    num_beams: int = Field(4, ge=1, le=8, description="Number of beams for beam search")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    do_sample: bool = Field(False, description="Use sampling instead of beam search")


class GeneratedReport(BaseModel):
    """Response model for generated report."""
    report: str = Field(..., description="Generated radiology report")
    findings: Optional[str] = Field(None, description="Extracted findings section")
    impression: Optional[str] = Field(None, description="Extracted impression section")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    confidence_score: Optional[float] = Field(None, description="Model confidence score")


class FeedbackRequest(BaseModel):
    """Request model for radiologist feedback."""
    original_report: str = Field(..., description="Original AI-generated report")
    corrected_report: str = Field(..., description="Radiologist-corrected report")
    image_id: Optional[str] = Field(None, description="Optional image identifier")
    feedback_notes: Optional[str] = Field(None, description="Additional feedback notes")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    success: bool
    message: str
    feedback_id: str


class ModelInfo(BaseModel):
    """Response model for model information."""
    model_name: str
    encoder: str
    decoder: str
    projection_type: str  # "HAQT-ARR" or "Standard"
    projection_queries: int
    anatomical_regions: Optional[List[str]] = None  # For HAQT-ARR
    total_parameters: int
    trainable_parameters: int
    device: str
    max_length: int


class AttentionVisualization(BaseModel):
    """Response model for attention visualization."""
    anatomical_regions: List[str]
    region_weights: List[float]
    spatial_priors: Optional[List[List[List[float]]]] = None  # [regions, H, W]
    generation_time_ms: float


# ============================================
# API Endpoints
# ============================================

@router.post("/generate", response_model=GeneratedReport)
async def generate_report(file: UploadFile = File(...)):
    """
    Generate a radiology report from an uploaded X-ray image.

    This endpoint accepts an X-ray image file and returns a generated
    clinical radiology report with findings and impressions.
    """
    from .main import get_model, get_device_instance

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image/*"
            )

        # Read and process image
        start_time = time.time()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Apply transforms
        transform = XRayTransform(get_val_transforms(384))
        image_tensor = transform(image).unsqueeze(0)

        # Get model and device
        model = get_model()
        device = get_device_instance()

        # Move to device
        image_tensor = image_tensor.to(device)

        # Generate report
        with torch.no_grad():
            _, generated_texts, _ = model.generate(
                images=image_tensor,
                max_length=256,
                num_beams=4,
                temperature=1.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        report = generated_texts[0]
        generation_time = (time.time() - start_time) * 1000

        # Parse report sections
        findings, impression = parse_report_sections(report)

        logger.info(f"Generated report in {generation_time:.2f}ms")

        return GeneratedReport(
            report=report,
            findings=findings,
            impression=impression,
            generation_time_ms=generation_time,
            confidence_score=None,  # TODO: Add confidence estimation
        )

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/base64", response_model=GeneratedReport)
async def generate_report_base64(request: ReportGenerationRequest):
    """
    Generate a radiology report from a base64-encoded X-ray image.

    Alternative endpoint that accepts base64-encoded images,
    useful for web applications.
    """
    from .main import get_model, get_device_instance

    try:
        # Decode base64 image
        start_time = time.time()

        # Remove data URL prefix if present
        image_data = request.image_base64
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply transforms
        transform = XRayTransform(get_val_transforms(384))
        image_tensor = transform(image).unsqueeze(0)

        # Get model and device
        model = get_model()
        device = get_device_instance()

        # Move to device
        image_tensor = image_tensor.to(device)

        # Generate report
        with torch.no_grad():
            _, generated_texts, _ = model.generate(
                images=image_tensor,
                max_length=request.max_length,
                num_beams=request.num_beams,
                temperature=request.temperature,
                do_sample=request.do_sample,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        report = generated_texts[0]
        generation_time = (time.time() - start_time) * 1000

        # Parse report sections
        findings, impression = parse_report_sections(report)

        logger.info(f"Generated report in {generation_time:.2f}ms")

        return GeneratedReport(
            report=report,
            findings=findings,
            impression=impression,
            generation_time_ms=generation_time,
            confidence_score=None,
        )

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/batch")
async def generate_reports_batch(files: List[UploadFile] = File(...)):
    """
    Generate reports for multiple X-ray images in batch.

    Processes multiple images in a single request for efficiency.
    """
    from .main import get_model, get_device_instance

    try:
        start_time = time.time()

        # Process all images
        transform = XRayTransform(get_val_transforms(384))
        image_tensors = []

        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for {file.filename}"
                )

            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_tensor = transform(image)
            image_tensors.append(image_tensor)

        # Stack into batch
        batch_tensor = torch.stack(image_tensors)

        # Get model and device
        model = get_model()
        device = get_device_instance()

        batch_tensor = batch_tensor.to(device)

        # Generate reports
        with torch.no_grad():
            _, generated_texts, _ = model.generate(
                images=batch_tensor,
                max_length=256,
                num_beams=4,
            )

        generation_time = (time.time() - start_time) * 1000

        # Build response
        results = []
        for i, (text, file) in enumerate(zip(generated_texts, files)):
            findings, impression = parse_report_sections(text)
            results.append({
                "filename": file.filename,
                "report": text,
                "findings": findings,
                "impression": impression,
            })

        return {
            "results": results,
            "total_images": len(files),
            "total_time_ms": generation_time,
            "avg_time_per_image_ms": generation_time / len(files),
        }

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit radiologist feedback for human-in-the-loop learning.

    Collects corrected reports from radiologists for model improvement.
    """
    import uuid
    from datetime import datetime

    try:
        # Generate feedback ID
        feedback_id = str(uuid.uuid4())[:8]

        # Store feedback (in production, save to database)
        feedback_data = {
            "id": feedback_id,
            "timestamp": datetime.utcnow().isoformat(),
            "original_report": feedback.original_report,
            "corrected_report": feedback.corrected_report,
            "image_id": feedback.image_id,
            "notes": feedback.feedback_notes,
        }

        # Save to file (in production, use database)
        feedback_dir = Path("data/feedback")
        feedback_dir.mkdir(parents=True, exist_ok=True)

        feedback_file = feedback_dir / f"{feedback_id}.json"
        import json
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)

        logger.info(f"Feedback saved: {feedback_id}")

        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            feedback_id=feedback_id,
        )

    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns model architecture details and configuration,
    including HAQT-ARR specific information if enabled.
    """
    from .main import get_model, get_device_instance

    try:
        model = get_model()
        device = get_device_instance()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Determine projection type and get anatomical regions
        if hasattr(model, 'use_anatomical_attention') and model.use_anatomical_attention:
            projection_type = "HAQT-ARR (Hierarchical Anatomical Query Tokens with Adaptive Region Routing)"
            anatomical_regions = model.get_anatomical_regions()
            num_queries = 8 + 7 * 4  # global + region queries
        else:
            projection_type = "Standard Multimodal Projection"
            anatomical_regions = None
            num_queries = 32

        return ModelInfo(
            model_name="XR2Text",
            encoder="Swin Transformer Base",
            decoder="BioBART",
            projection_type=projection_type,
            projection_queries=num_queries,
            anatomical_regions=anatomical_regions,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            device=str(device),
            max_length=256,
        )

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attention/visualize", response_model=AttentionVisualization)
async def visualize_attention(file: UploadFile = File(...)):
    """
    Visualize HAQT-ARR attention patterns for an X-ray image.

    Returns anatomical region weights and spatial priors (Novel).
    Only available when HAQT-ARR is enabled.
    """
    from .main import get_model, get_device_instance

    try:
        # Check if HAQT-ARR is enabled
        model = get_model()
        device = get_device_instance()

        if not (hasattr(model, 'use_anatomical_attention') and model.use_anatomical_attention):
            raise HTTPException(
                status_code=400,
                detail="HAQT-ARR not enabled. Attention visualization requires anatomical attention."
            )

        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type")

        start_time = time.time()

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Apply transforms
        transform = XRayTransform(get_val_transforms(384))
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get attention visualization
        with torch.no_grad():
            attn_data = model.get_attention_visualization(image_tensor)

        generation_time = (time.time() - start_time) * 1000

        # Extract data
        region_names = attn_data.get('anatomical_regions', [])
        region_weights = attn_data.get('region_weights')
        spatial_priors = attn_data.get('spatial_priors')

        # Convert to lists
        region_weights_list = region_weights[0].cpu().tolist() if region_weights is not None else []
        spatial_priors_list = spatial_priors.cpu().tolist() if spatial_priors is not None else None

        return AttentionVisualization(
            anatomical_regions=region_names,
            region_weights=region_weights_list,
            spatial_priors=spatial_priors_list,
            generation_time_ms=generation_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Attention visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/status")
async def get_model_status():
    """
    Get current model status and GPU usage.
    """
    from .main import get_model, get_device_instance

    try:
        model = get_model()
        device = get_device_instance()

        status = {
            "model_loaded": True,
            "device": str(device),
            "mode": "eval" if not model.training else "train",
        }

        if torch.cuda.is_available():
            status["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            }

        return status

    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Helper Functions
# ============================================

def parse_report_sections(report: str) -> tuple:
    """
    Parse a generated report into findings and impression sections.

    Args:
        report: Full generated report text

    Returns:
        Tuple of (findings, impression) strings
    """
    findings = None
    impression = None

    report_upper = report.upper()

    if "FINDINGS:" in report_upper:
        parts = report.split("FINDINGS:", 1)
        if len(parts) > 1:
            rest = parts[1]
            if "IMPRESSION:" in rest.upper():
                findings_part, impression_part = rest.upper().split("IMPRESSION:", 1)
                findings = rest[:len(findings_part)].strip()
                impression = rest[len(findings_part) + len("IMPRESSION:"):].strip()
            else:
                findings = rest.strip()

    elif "IMPRESSION:" in report_upper:
        parts = report.split("IMPRESSION:", 1)
        if len(parts) > 1:
            impression = parts[1].strip()

    else:
        # No explicit sections, treat entire report as findings
        findings = report.strip()

    return findings, impression
