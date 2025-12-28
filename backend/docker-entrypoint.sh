#!/bin/bash
set -e

echo "========================================"
echo "CityWok Audio Fingerprinting API"
echo "========================================"
echo ""

# Configuration
S3_BUCKET="${S3_BUCKET:-citywok-audio-db}"
S3_PREFIX="${S3_PREFIX:-fingerprints}"
DB_VERSION="${DB_VERSION:-v1}"
DATA_DIR="${DATA_DIR:-/app/data}"
FINGERPRINT_DIR="${DATA_DIR}/fingerprints/${DB_VERSION}"
USE_LMDB="${USE_LMDB:-false}"

echo "Configuration:"
echo "  USE_LMDB: $USE_LMDB"
echo "  DATA_DIR: $DATA_DIR"
echo "  FINGERPRINT_DIR: $FINGERPRINT_DIR"
echo ""

# Check if LMDB mode is enabled
if [ "$USE_LMDB" = "true" ] || [ "$USE_LMDB" = "1" ]; then
    echo "🔧 LMDB mode enabled"

    # Check if databases need to be downloaded
    if [ ! -d "$FINGERPRINT_DIR" ] || [ -z "$(ls -A $FINGERPRINT_DIR 2>/dev/null)" ]; then
        echo "📦 Fingerprint databases not found locally"

        # Check if S3 credentials are configured
        if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
            echo "   Downloading from S3..."
            echo "   Bucket: s3://${S3_BUCKET}/${S3_PREFIX}/${DB_VERSION}/"
            echo ""

            mkdir -p "$FINGERPRINT_DIR"

            # Use Python to download databases (only seasons 1-5 for local testing)
            MAX_SEASONS="${MAX_SEASONS:-5}"
            python3 -c "
from app.core.storage import storage
import sys
import os

max_seasons = int(os.getenv('MAX_SEASONS', '5'))
success_count = storage.download_all_seasons(
    '${FINGERPRINT_DIR}',
    seasons=range(1, max_seasons + 1),
    verify_checksums=True
)

if success_count == 0:
    print('❌ Failed to download any databases')
    sys.exit(1)
else:
    print(f'✓ Downloaded {success_count} season databases')
" || {
                echo "❌ Database download failed!"
                echo "   Falling back to empty database (will need to ingest episodes)"
            }
        else
            echo "   ⚠️  S3 credentials not configured"
            echo "   Falling back to empty database (will need to ingest episodes)"
            echo ""
            echo "   To download databases from S3, set:"
            echo "     AWS_ACCESS_KEY_ID"
            echo "     AWS_SECRET_ACCESS_KEY"
            echo "     S3_BUCKET (default: citywok-audio-db)"
            echo "     AWS_ENDPOINT_URL (for Cloudflare R2)"
        fi
    else
        echo "✓ Fingerprint databases found locally"

        # Count databases
        db_count=$(ls -d ${FINGERPRINT_DIR}/season_*.lmdb 2>/dev/null | wc -l)
        echo "  Found $db_count season databases"
    fi
else
    echo "🔧 Pickle mode enabled (legacy)"
    
    # Check if pickle databases need to be downloaded from R2/S3
    # Only download if files don't already exist (volume persistence)
    pickle_count=$(ls -1 ${DATA_DIR}/audio_fingerprints_s*.pkl 2>/dev/null | wc -l | tr -d ' ')
    
    if [ -n "$DOWNLOAD_PICKLE_DB" ] && [ "$DOWNLOAD_PICKLE_DB" = "true" ]; then
        if [ "$pickle_count" -lt 20 ]; then
            if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
                echo "📦 Downloading missing pickle databases from R2/S3..."
                echo "   Found $pickle_count existing files, downloading missing ones..."
                
                mkdir -p "$DATA_DIR"
                
                # Download pickle files from R2/S3
                python3 -c "
import os
import boto3
from pathlib import Path

bucket = os.getenv('S3_BUCKET', 'citywok-audio-db')
prefix = os.getenv('S3_PREFIX', 'pickle')
endpoint = os.getenv('AWS_ENDPOINT_URL')
data_dir = os.getenv('DATA_DIR', '/app/data')

s3 = boto3.client(
    's3',
    endpoint_url=endpoint,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='auto'
)

# List and download pickle files (skip if already exists)
try:
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + '/')
    if 'Contents' in response:
        downloaded = 0
        skipped = 0
        for obj in response['Contents']:
            key = obj['Key']
            filename = key.split('/')[-1]
            if filename.endswith('.pkl'):
                local_path = os.path.join(data_dir, filename)
                if os.path.exists(local_path):
                    skipped += 1
                    continue
                print(f'  Downloading {filename}...', end=' ', flush=True)
                s3.download_file(bucket, key, local_path)
                print('✓')
                downloaded += 1
        if downloaded > 0:
            print(f'✓ Downloaded {downloaded} pickle database files')
        if skipped > 0:
            print(f'⏭️  Skipped {skipped} files (already exist)')
    else:
        print('  No pickle files found in R2')
except Exception as e:
    print(f'  ⚠️  Could not download from R2: {e}')
" || echo "   ⚠️  Pickle download failed, continuing without databases"
            else
                echo "   ⚠️  R2 credentials not configured for pickle download"
                echo "   (Found $pickle_count/20 files - missing files will not be downloaded)"
            fi
        else
            echo "   ✓ All pickle databases already present ($pickle_count files)"
        fi
    fi
    
    # Check if pickle files exist locally
    pickle_count=$(ls -1 ${DATA_DIR}/audio_fingerprints_s*.pkl 2>/dev/null | wc -l | tr -d ' ')
    if [ "$pickle_count" -gt 0 ]; then
        echo "   ✓ Found $pickle_count pickle database file(s)"
    else
        echo "   ⚠️  No pickle databases found (will need to ingest episodes)"
    fi
fi

echo ""
echo "========================================"
echo "Starting API server..."
echo "========================================"
echo ""

# Start the application
exec "$@"
