import os

class Settings:
    PROJECT_NAME: str = "CityWok"
    API_V1_STR: str = "/api/v1"
    
    # Storage - can be overridden with environment variable
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
    UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
    
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    S3_PREFIX: str = os.getenv("S3_PREFIX", "audio_db/")

settings = Settings()
