#!/usr/bin/env python3
"""
Upload pickle database files to S3/R2.
Supports large files (>300MB) that can't be uploaded via web interface.

Usage:
    # Upload a single season
    python upload_pickle_season.py 1

    # Upload multiple seasons
    python upload_pickle_season.py 1 2 3 4 5 6 7 15

    # Upload all seasons
    python upload_pickle_season.py --all
"""

import os
import sys
import boto3
from pathlib import Path

# Try to import tqdm for progress bar, fallback to simple upload if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def upload_file_with_progress(s3_client, bucket, key, filepath):
    """Upload file with progress bar if available."""
    if HAS_TQDM:
        file_size = os.path.getsize(filepath)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            def callback(bytes_amount):
                pbar.update(bytes_amount)
            
            s3_client.upload_file(
                filepath,
                bucket,
                key,
                Callback=callback
            )
    else:
        # Simple upload without progress bar
        s3_client.upload_file(filepath, bucket, key)

def upload_season_pickle(season: int, data_dir: str = None):
    """Upload a specific season's pickle file to S3/R2."""
    
    # Get configuration from environment
    bucket = os.getenv('S3_BUCKET', 'citywok-audio-db')
    prefix = os.getenv('S3_PREFIX', 'pickle')
    endpoint = os.getenv('AWS_ENDPOINT_URL')
    
    if data_dir is None:
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
        
        # Test credentials by listing bucket
        print(f"🔍 Testing credentials...")
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"✅ Credentials valid, bucket accessible")
        except Exception as e:
            print(f"❌ Credential test failed: {e}")
            print(f"   Make sure you're using R2 API Token credentials, not account ID")
            return False
    except Exception as e:
        print(f"❌ Error creating S3 client: {e}")
        return False
    
    # Construct file paths
    filename = f"audio_fingerprints_s{season:02d}.pkl"
    s3_key = f"{prefix}/{filename}"
    local_path = os.path.join(data_dir, filename)
    
    # Check if file exists locally
    if not os.path.exists(local_path):
        print(f"❌ File not found: {local_path}")
        return False
    
    file_size = os.path.getsize(local_path) / (1024 * 1024)
    print(f"📤 Uploading {filename} ({file_size:.1f} MB) to s3://{bucket}/{s3_key}...")
    
    # Check if file already exists in R2
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        print(f"⚠️  File already exists in R2")
        response = input("   Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipping upload")
            return True
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            print(f"⚠️  Error checking existing file: {e}")
            # Continue anyway
    
    # Upload file
    try:
        upload_file_with_progress(s3, bucket, s3_key, local_path)
        
        print(f"✅ Uploaded {filename} successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        return False

def upload_all_seasons(data_dir: str = None):
    """Upload all existing pickle files."""
    if data_dir is None:
        data_dir = os.getenv('DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))
    
    # Find all pickle files
    pickle_files = sorted(Path(data_dir).glob("audio_fingerprints_s*.pkl"))
    
    if not pickle_files:
        print(f"❌ No pickle files found in {data_dir}")
        return False
    
    print(f"📦 Found {len(pickle_files)} pickle files to upload\n")
    
    seasons = []
    for pkl_file in pickle_files:
        # Extract season number from filename
        match = __import__('re').search(r's(\d+)\.pkl', pkl_file.name)
        if match:
            seasons.append(int(match.group(1)))
    
    seasons = sorted(seasons)
    print(f"Seasons to upload: {seasons}\n")
    
    successful = 0
    failed = 0
    
    for season in seasons:
        if upload_season_pickle(season, data_dir):
            successful += 1
        else:
            failed += 1
        print()  # Blank line between uploads
    
    print("="*60)
    print(f"Upload Summary: {successful} successful, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        success = upload_all_seasons()
        sys.exit(0 if success else 1)
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Get data directory
    data_dir = os.getenv('DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))
    
    # Upload specified seasons
    successful = 0
    failed = 0
    
    for arg in sys.argv[1:]:
        try:
            season = int(arg)
            if season < 1 or season > 20:
                print(f"⚠️  Season {season} out of range (1-20), skipping")
                failed += 1
                continue
            
            if upload_season_pickle(season, data_dir):
                successful += 1
            else:
                failed += 1
            print()  # Blank line between uploads
        except ValueError:
            print(f"⚠️  Invalid season number: {arg}, skipping")
            failed += 1
    
    if successful > 0 or failed > 0:
        print("="*60)
        print(f"Upload Summary: {successful} successful, {failed} failed")
        print("="*60)
    
    sys.exit(0 if failed == 0 else 1)

