import os

class Settings:
    PROJECT_NAME: str = "CityWok"
    API_V1_STR: str = "/api/v1"
    
    # Video Processing
    FRAME_SAMPLING_RATE: int = 1  # Frames per second
    
    # Model
    CLIP_MODEL_ID: str = "openai/clip-vit-base-patch32"
    
    # Storage
    DATA_DIR: str = os.path.join(os.getcwd(), "data")
    UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
    VECTOR_DB_PATH: str = os.path.join(DATA_DIR, "vector_index.faiss")
    METADATA_DB_PATH: str = os.path.join(DATA_DIR, "metadata.json")

settings = Settings()


