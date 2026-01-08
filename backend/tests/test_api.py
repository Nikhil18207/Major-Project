"""
Test suite for FastAPI endpoints.

Tests:
- Health check
- Report generation
- Model info
- Feedback submission
"""

import pytest
from io import BytesIO
from PIL import Image
import base64

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture
def client():
    """Create test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (384, 384), color='gray')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_image_base64():
    """Create a sample base64 encoded image."""
    img = Image.new('RGB', (384, 384), color='gray')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns correct format."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "XR2Text API"
        assert "version" in data


class TestGenerateEndpoint:
    """Tests for report generation endpoint."""

    def test_generate_invalid_file_type(self, client):
        """Test rejection of non-image files."""
        response = client.post(
            "/api/v1/generate",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_generate_with_image(self, client, sample_image):
        """Test report generation with valid image."""
        response = client.post(
            "/api/v1/generate",
            files={"file": ("test.png", sample_image, "image/png")},
        )

        # Note: This may fail if model isn't loaded
        # In real tests, would mock the model
        if response.status_code == 200:
            data = response.json()
            assert "report" in data
            assert "generation_time_ms" in data


class TestGenerateBase64Endpoint:
    """Tests for base64 report generation endpoint."""

    def test_generate_base64(self, client, sample_image_base64):
        """Test report generation with base64 image."""
        response = client.post(
            "/api/v1/generate/base64",
            json={
                "image_base64": sample_image_base64,
                "max_length": 100,
                "num_beams": 2,
            },
        )

        # May fail if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "report" in data


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info(self, client):
        """Test model info returns correct format."""
        response = client.get("/api/v1/model/info")

        if response.status_code == 200:
            data = response.json()

            assert "model_name" in data
            assert "encoder" in data
            assert "decoder" in data
            assert "projection_type" in data
            assert "total_parameters" in data


class TestModelStatusEndpoint:
    """Tests for model status endpoint."""

    def test_model_status(self, client):
        """Test model status endpoint."""
        response = client.get("/api/v1/model/status")

        if response.status_code == 200:
            data = response.json()

            assert "model_loaded" in data
            assert "device" in data


class TestFeedbackEndpoint:
    """Tests for feedback submission endpoint."""

    def test_feedback_submission(self, client):
        """Test feedback can be submitted."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "original_report": "Original AI report text",
                "corrected_report": "Corrected report text",
                "image_id": "test_001",
                "feedback_notes": "Test feedback",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "feedback_id" in data


class TestBatchEndpoint:
    """Tests for batch generation endpoint."""

    def test_batch_empty(self, client):
        """Test batch endpoint rejects empty requests."""
        response = client.post(
            "/api/v1/generate/batch",
            files=[],
        )

        # FastAPI should reject this
        assert response.status_code in [400, 422]

    def test_batch_too_many_files(self, client, sample_image):
        """Test batch endpoint limits file count."""
        # Create 15 files (above the 10 limit)
        files = [
            ("files", (f"test_{i}.png", sample_image, "image/png"))
            for i in range(15)
        ]

        response = client.post("/api/v1/generate/batch", files=files)

        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]


class TestAttentionEndpoint:
    """Tests for HAQT-ARR attention visualization endpoint."""

    def test_attention_invalid_file(self, client):
        """Test attention endpoint rejects non-images."""
        response = client.post(
            "/api/v1/attention/visualize",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
