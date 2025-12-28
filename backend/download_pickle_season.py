#!/usr/bin/env python3
"""
Download a specific season's pickle database from S3/R2.
Usage: python download_pickle_season.py <season_number>
Example: python download_pickle_season.py 4
"""

import os
import sys
import boto3
from pathlib import Path

def download_season_pickle(season: int):
    """Download a specific season's pickle file from S3/R2."""
    
    # Get configuration from environment
    bucket = os.getenv('S3_BUCKET', 'citywok-audio-db')
    prefix = os.getenv('S3_PREFIX', 'pickle')
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    data_dir = os.getenv('DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))
    
    # Check for credentials
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not access_key or not secret_key:
        print("❌ Error: AWS credentials not found!")
        print("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        return False
    
    # Create S3 client
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto'
        )
    except Exception as e:
        print(f"❌ Error creating S3 client: {e}")
        return False
    
    # Construct file paths
    filename = f"audio_fingerprints_s{season:02d}.pkl"
    s3_key = f"{prefix}/{filename}"
    local_path = os.path.join(data_dir, filename)
    
    # Check if already exists
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"ℹ️  {filename} already exists ({file_size:.1f} MB)")
        response = input("   Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipping download")
            return True
    
    # Create data directory if needed
    os.makedirs(data_dir, exist_ok=True)
    
    # Download file
    try:
        print(f"📥 Downloading {filename} from s3://{bucket}/{s3_key}...")
        print(f"   Destination: {local_path}")
        
        s3.download_file(bucket, s3_key, local_path)
        
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ Downloaded {filename} ({file_size:.1f} MB)")
        return True
        
    except s3.exceptions.NoSuchKey:
        print(f"❌ File not found: s3://{bucket}/{s3_key}")
        print(f"   Check that the file exists in your S3/R2 bucket")
        return False
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        # Clean up partial file
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_pickle_season.py <season_number>")
        print("Example: python download_pickle_season.py 4")
        sys.exit(1)
    
    try:
        season = int(sys.argv[1])
        if season < 1 or season > 20:
            print("❌ Season must be between 1 and 20")
            sys.exit(1)
    except ValueError:
        print("❌ Invalid season number")
        sys.exit(1)
    
    success = download_season_pickle(season)
    sys.exit(0 if success else 1)

