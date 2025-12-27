# Backend Utility Scripts

This directory contains utility scripts for managing the CityWok audio fingerprint database.

## Directory Structure

- **ingestion/** - Scripts for ingesting audio from video files
- **storage/** - Scripts for uploading/downloading databases to/from S3/R2
- **database/** - Database inspection and maintenance tools
- **testing/** - Test suites

## Quick Reference

### Ingestion Scripts

All scripts should be run from the `backend/` directory:

```bash
cd backend

# Ingest a single season
./scripts/ingest_season.sh 6

# Ingest multiple seasons
./scripts/ingest_all_seasons.sh 1 15

# Offline LMDB ingestion
python scripts/ingestion/offline_ingest.py "/path/to/Season 1" 1

# Ingest missing episodes
python scripts/ingestion/ingest_missing_seasons.py
```

**Environment Variables:**
- `EPISODES_DIR` - Base directory containing Season folders (default: /Users/pravinlohani/Downloads)

**Examples:**
```bash
# Use custom episodes directory
EPISODES_DIR=/mnt/media/southpark ./scripts/ingest_season.sh 10
```

### Storage Scripts

```bash
# Upload a season's pickle database to S3/R2
python scripts/storage/upload_pickle_season.py 1

# Upload multiple seasons
python scripts/storage/upload_pickle_season.py 1 2 3 4 5

# Upload all seasons
python scripts/storage/upload_pickle_season.py --all

# Download a specific season
python scripts/storage/download_pickle_season.py 5
```

### Database Tools

```bash
# Check which episodes are in the database
python scripts/database/check_db.py
```

### Testing

```bash
# Run tests via make
make test

# Or directly with pytest
pytest scripts/testing/test_season_matching.py -v
```

## Requirements

All scripts require the backend virtual environment to be activated:

```bash
cd backend
source venv/bin/activate
```

## Storage Configuration

For S3/R2 operations, set these environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_ENDPOINT_URL="https://your-r2-endpoint"  # For Cloudflare R2
export S3_BUCKET="citywok-audio-db"
export S3_PREFIX="fingerprints"
```
