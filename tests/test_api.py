"""Tests for FastAPI server."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from src.api.server import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns valid response."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestAuthenticatedEndpoints:
    """Test endpoints requiring authentication."""

    def test_predict_without_key(self, client):
        """Test prediction fails without API key."""
        response = client.post("/predict")
        assert response.status_code == 401

    def test_predict_with_invalid_key(self, client):
        """Test prediction fails with invalid API key."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 403

    def test_price_without_key(self, client):
        """Test price endpoint fails without API key."""
        response = client.get("/price")
        assert response.status_code == 401


class TestPredictionEndpoint:
    """Test prediction endpoint."""

    def test_predict_with_valid_key(self, client):
        """Test prediction with valid API key."""
        # Note: This will fail if models aren't trained
        # In a real test environment, you'd mock the predictor
        response = client.post(
            "/predict",
            headers={"X-API-Key": "dev-key-change-in-production"},
        )
        # Either success (200) or service unavailable (503) if models not trained
        assert response.status_code in [200, 503]
