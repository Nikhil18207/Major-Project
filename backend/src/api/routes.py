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
from ..models.anatomical_attention import NUM_ANATOMICAL_REGIONS


router = APIRouter()

# ============================================
# Configuration Constants
# ============================================
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_BATCH_SIZE = 10  # Maximum images in batch request
ALLOWED_CONTENT_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"]


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

    # NOVEL: Enhanced response fields
    reliability: Optional[str] = Field(None, description="Reliability category: high/medium/low")
    needs_review: Optional[bool] = Field(None, description="Whether radiologist review is recommended")
    uncertainty: Optional[Dict] = Field(None, description="Detailed uncertainty metrics")
    grounding: Optional[Dict] = Field(None, description="Factual grounding validation")
    explanation: Optional[Dict] = Field(None, description="Explainability information")


class AnalyzedReport(BaseModel):
    """Response model for comprehensive report analysis (Novel)."""
    report: str = Field(..., description="Generated radiology report")
    findings: Optional[str] = Field(None, description="Extracted findings section")
    impression: Optional[str] = Field(None, description="Extracted impression section")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")

    # Uncertainty (Novel)
    confidence_score: float = Field(..., description="Overall confidence (0-1)")
    epistemic_uncertainty: float = Field(..., description="Model uncertainty")
    aleatoric_uncertainty: float = Field(..., description="Data uncertainty")
    reliability: str = Field(..., description="Reliability: high/medium/low")
    needs_review: bool = Field(..., description="Radiologist review recommended")
    ood_score: float = Field(..., description="Out-of-distribution score (0-1)")

    # Grounding (Novel)
    is_grounded: bool = Field(..., description="Whether findings are grounded in image")
    hallucination_score: float = Field(..., description="Hallucination risk (0-1)")
    consistency_violations: List[str] = Field(default=[], description="Medical consistency issues")

    # Explanations (Novel)
    finding_explanations: List[Dict] = Field(default=[], description="Per-finding explanations")
    attention_summary: Dict[str, float] = Field(default={}, description="Attention by region")
    key_observations: List[str] = Field(default=[], description="Key clinical observations")
    recommendations: List[str] = Field(default=[], description="Suggested actions")


class FeedbackRequest(BaseModel):
    """Request model for radiologist feedback with validation."""
    original_report: str = Field(
        ...,
        description="Original AI-generated report",
        min_length=1,
        max_length=10000,
    )
    corrected_report: str = Field(
        ...,
        description="Radiologist-corrected report",
        min_length=1,
        max_length=10000,
    )
    image_id: Optional[str] = Field(
        None,
        description="Optional image identifier",
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\-\.]*$',  # Only alphanumeric, underscore, hyphen, dot
    )
    feedback_notes: Optional[str] = Field(
        None,
        description="Additional feedback notes",
        max_length=5000,
    )


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
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
            )

        # Read and validate file size
        start_time = time.time()
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
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

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

        # Validate decoded image size (same as file upload)
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Maximum size is {MAX_FILE_SIZE_MB}MB"
            )

        # Validate image can be opened
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        # Validate image dimensions (sanity check)
        if image.width < 32 or image.height < 32:
            raise HTTPException(status_code=400, detail="Image too small. Minimum 32x32 pixels.")
        if image.width > 8192 or image.height > 8192:
            raise HTTPException(status_code=400, detail="Image too large. Maximum 8192x8192 pixels.")

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


@router.post("/generate/analyze", response_model=AnalyzedReport)
async def generate_report_with_analysis(file: UploadFile = File(...)):
    """
    NOVEL: Generate report with comprehensive analysis.

    This endpoint provides:
    - Generated radiology report
    - Uncertainty quantification (confidence, epistemic/aleatoric)
    - Factual grounding validation (hallucination detection)
    - Detailed explanations (evidence regions, reasoning)
    - Clinical recommendations

    Use this endpoint for production clinical decision support.
    """
    from .main import get_model, get_device_instance

    try:
        # Validate file type
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}"
            )

        start_time = time.time()
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        transform = XRayTransform(get_val_transforms(384))
        image_tensor = transform(image).unsqueeze(0)

        model = get_model()
        device = get_device_instance()
        image_tensor = image_tensor.to(device)

        # Generate with comprehensive analysis
        with torch.no_grad():
            analysis = model.generate_with_analysis(
                images=image_tensor,
                include_uncertainty=True,
                include_grounding=True,
                include_explanation=True,
                include_auxiliary=True,
                max_length=256,
                num_beams=4,
                temperature=1.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        report = analysis["reports"][0]
        generation_time = (time.time() - start_time) * 1000
        findings, impression = parse_report_sections(report)

        # Extract uncertainty info
        uncertainty = analysis.get("uncertainty", {})
        grounding = analysis.get("grounding", {})
        explanation = analysis.get("explanation", {})

        logger.info(
            f"Generated analyzed report in {generation_time:.2f}ms | "
            f"Confidence: {uncertainty.get('confidence_score', 0):.2f} | "
            f"Grounded: {grounding.get('is_grounded', True)}"
        )

        return AnalyzedReport(
            report=report,
            findings=findings,
            impression=impression,
            generation_time_ms=generation_time,
            # Uncertainty
            confidence_score=uncertainty.get("confidence_score", 0.7),
            epistemic_uncertainty=uncertainty.get("epistemic_uncertainty", 0.3),
            aleatoric_uncertainty=uncertainty.get("aleatoric_uncertainty", 0.3),
            reliability=uncertainty.get("reliability", "medium"),
            needs_review=uncertainty.get("needs_review", True),
            ood_score=uncertainty.get("ood_score", 0.0),
            # Grounding
            is_grounded=grounding.get("is_grounded", True),
            hallucination_score=grounding.get("hallucination_score", 0.0),
            consistency_violations=grounding.get("consistency_violations", []),
            # Explanations
            finding_explanations=explanation.get("finding_explanations", []),
            attention_summary=explanation.get("attention_summary", {}),
            key_observations=explanation.get("key_observations", []),
            recommendations=explanation.get("recommendations", []),
        )

    except Exception as e:
        logger.error(f"Analyzed report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/batch")
async def generate_reports_batch(files: List[UploadFile] = File(...)):
    """
    Generate reports for multiple X-ray images in batch.

    Processes multiple images in a single request for efficiency.
    Maximum {MAX_BATCH_SIZE} images per request.
    """
    from .main import get_model, get_device_instance

    try:
        # Validate batch size
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum batch size: {MAX_BATCH_SIZE}"
            )

        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")

        start_time = time.time()

        # Process all images
        transform = XRayTransform(get_val_transforms(384))
        image_tensors = []

        for file in files:
            if file.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for {file.filename}: {file.content_type}"
                )

            contents = await file.read()

            # Validate file size
            if len(contents) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} too large. Maximum: {MAX_FILE_SIZE_MB}MB"
                )
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


def sanitize_text(text: str) -> str:
    """
    Sanitize text input to prevent injection attacks.

    Removes or escapes potentially dangerous characters while
    preserving medical terminology and formatting.
    """
    if text is None:
        return None

    # Remove null bytes and other control characters (except newlines and tabs)
    import re
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Limit consecutive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', '  ', text)

    return text.strip()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit radiologist feedback for human-in-the-loop learning.

    Collects corrected reports from radiologists for model improvement.
    Input is validated via Pydantic and sanitized before storage.
    """
    import uuid
    import json
    import re
    from datetime import datetime

    try:
        # Generate secure feedback ID (full UUID for uniqueness)
        feedback_id = str(uuid.uuid4())

        # Sanitize all text inputs
        sanitized_original = sanitize_text(feedback.original_report)
        sanitized_corrected = sanitize_text(feedback.corrected_report)
        sanitized_notes = sanitize_text(feedback.feedback_notes) if feedback.feedback_notes else None

        # Validate image_id format (additional check beyond Pydantic)
        sanitized_image_id = None
        if feedback.image_id:
            # Only allow safe characters in image_id
            if re.match(r'^[a-zA-Z0-9_\-\.]+$', feedback.image_id):
                sanitized_image_id = feedback.image_id[:100]  # Truncate to max length
            else:
                logger.warning(f"Invalid image_id format rejected: {feedback.image_id[:20]}...")

        # Store feedback with sanitized data
        feedback_data = {
            "id": feedback_id,
            "timestamp": datetime.utcnow().isoformat(),
            "original_report": sanitized_original,
            "corrected_report": sanitized_corrected,
            "image_id": sanitized_image_id,
            "notes": sanitized_notes,
            "metadata": {
                "original_length": len(feedback.original_report),
                "corrected_length": len(feedback.corrected_report),
            }
        }

        # Secure file path handling - prevent path traversal
        feedback_dir = Path("data/feedback").resolve()
        feedback_dir.mkdir(parents=True, exist_ok=True)

        # Use only the UUID (already validated) as filename
        safe_filename = f"{feedback_id}.json"
        feedback_file = feedback_dir / safe_filename

        # Verify the file path is within the feedback directory (prevent traversal)
        if not str(feedback_file.resolve()).startswith(str(feedback_dir)):
            raise HTTPException(status_code=400, detail="Invalid file path")

        # Write with explicit encoding
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Feedback saved: {feedback_id[:8]}...")

        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            feedback_id=feedback_id[:8],  # Return shortened ID to user
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")


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
            # Calculate total queries: global queries + (num_regions * queries_per_region)
            # Uses NUM_ANATOMICAL_REGIONS constant instead of hardcoding 7
            num_global = 8
            num_per_region = 4
            num_queries = num_global + NUM_ANATOMICAL_REGIONS * num_per_region
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
    import re

    findings = None
    impression = None

    # Use case-insensitive regex for robust parsing
    # This handles variations like "FINDINGS:", "Findings:", "findings:"
    findings_pattern = re.compile(r'\bFINDINGS\s*:', re.IGNORECASE)
    impression_pattern = re.compile(r'\bIMPRESSION\s*:', re.IGNORECASE)

    findings_match = findings_pattern.search(report)
    impression_match = impression_pattern.search(report)

    if findings_match and impression_match:
        # Both sections present
        if findings_match.start() < impression_match.start():
            # FINDINGS comes before IMPRESSION
            findings_start = findings_match.end()
            findings_end = impression_match.start()
            impression_start = impression_match.end()

            findings = report[findings_start:findings_end].strip()
            impression = report[impression_start:].strip()
        else:
            # IMPRESSION comes before FINDINGS (unusual but handle it)
            impression_start = impression_match.end()
            impression_end = findings_match.start()
            findings_start = findings_match.end()

            impression = report[impression_start:impression_end].strip()
            findings = report[findings_start:].strip()

    elif findings_match:
        # Only FINDINGS section
        findings = report[findings_match.end():].strip()

    elif impression_match:
        # Only IMPRESSION section
        impression = report[impression_match.end():].strip()

    else:
        # No explicit sections, treat entire report as findings
        findings = report.strip()

    return findings, impression
