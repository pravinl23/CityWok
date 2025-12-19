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
    echo "   LMDB databases will not be loaded"
fi

echo ""
echo "========================================"
echo "Starting API server..."
echo "========================================"
echo ""

# Start the application
exec "$@"
