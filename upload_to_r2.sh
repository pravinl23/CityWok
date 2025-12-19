#!/bin/bash
# Upload pickle database files to Cloudflare R2

set -e

echo "📦 Uploading pickle files to Cloudflare R2"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Install it with: brew install awscli"
    exit 1
fi

# Check if credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_ENDPOINT_URL" ]; then
    echo "❌ Missing R2 credentials!"
    echo ""
    echo "Set these environment variables:"
    echo "  export AWS_ACCESS_KEY_ID='your_access_key'"
    echo "  export AWS_SECRET_ACCESS_KEY='your_secret_key'"
    echo "  export AWS_ENDPOINT_URL='https://your-account-id.r2.cloudflarestorage.com'"
    echo ""
    echo "Get these from Cloudflare R2 → Manage R2 API Tokens"
    exit 1
fi

BUCKET="${S3_BUCKET:-citywok-audio-db}"
PREFIX="${S3_PREFIX:-pickle}"

echo "Configuration:"
echo "  Bucket: $BUCKET"
echo "  Prefix: $PREFIX"
echo "  Endpoint: $AWS_ENDPOINT_URL"
echo ""

cd "$(dirname "$0")/backend/data"

# Count files
file_count=$(ls -1 audio_fingerprints_s*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "Found $file_count database files"
echo ""

# Upload each file
uploaded=0
failed=0

for file in audio_fingerprints_s*.pkl; do
    if [ -f "$file" ]; then
        file_size=$(du -h "$file" | cut -f1)
        echo -n "Uploading $file ($file_size)... "
        
        if aws s3 cp "$file" "s3://${BUCKET}/${PREFIX}/$file" \
            --endpoint-url="$AWS_ENDPOINT_URL" \
            --quiet 2>/dev/null; then
            echo "✅"
            ((uploaded++))
        else
            echo "❌"
            ((failed++))
        fi
    fi
done

echo ""
echo "=========================================="
echo "Upload Summary"
echo "=========================================="
echo "  ✅ Uploaded: $uploaded files"
echo "  ❌ Failed: $failed files"
echo ""
echo "Next: Configure Railway with R2 credentials and restart service"
echo "=========================================="

