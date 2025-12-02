from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app
import pytest
import numpy as np
from PIL import Image
import io

client = TestClient(app)

@pytest.fixture
def mock_video_processor():
    with patch("app.api.endpoints.video_processor") as mock:
        yield mock

@pytest.fixture
def mock_vector_db():
    with patch("app.api.endpoints.vector_db") as mock:
        yield mock

@pytest.fixture
def mock_audio_matcher():
    with patch("app.api.endpoints.audio_matcher") as mock:
        yield mock

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to CityWok API"}

def test_identify_invalid_file_type():
    response = client.post(
        "/api/v1/identify",
        files={"file": ("test.txt", b"content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

def test_identify_success(mock_video_processor, mock_vector_db, mock_audio_matcher):
    # Setup mocks
    mock_video_processor.extract_frames.return_value = [
        (1.0, Image.new('RGB', (10, 10))),
        (2.0, Image.new('RGB', (10, 10)))
    ]
    mock_video_processor.compute_embeddings.return_value = np.random.rand(2, 512)
    
    # Mock search results: 2 frames, 5 matches each
    # Frame 1 matches Ep1 at t=10
    # Frame 2 matches Ep1 at t=11
    mock_vector_db.search.return_value = [
        [(0.9, {"episode_id": "S01E01", "timestamp": 10.0})],
        [(0.9, {"episode_id": "S01E01", "timestamp": 11.0})]
    ]
    
    mock_audio_matcher.match_clip.return_value = {}
    
    # Create dummy video content
    file_content = b"fake video content"
    
    response = client.post(
        "/api/v1/identify",
        files={"file": ("test.mp4", file_content, "video/mp4")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["match_found"] is True
    assert data["episode"] == "S01E01"
    # Estimated start: Frame 1 (t=1) matches Ep (t=10) -> start = 9.0
    # Frame 2 (t=2) matches Ep (t=11) -> start = 9.0
    # 9.0 seconds = 0:00:09
    assert data["timestamp"] == "0:00:09"

def test_identify_no_match(mock_video_processor, mock_vector_db, mock_audio_matcher):
    mock_video_processor.extract_frames.return_value = []
    
    response = client.post(
        "/api/v1/identify",
        files={"file": ("test.mp4", b"content", "video/mp4")}
    )
    
    # If no frames, it returns no match
    assert response.status_code == 200
    data = response.json()
    assert data["match_found"] is False


