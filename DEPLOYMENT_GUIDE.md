# CityWok Cloud Deployment Guide 🚀

Complete guide to deploying CityWok audio fingerprinting backend to the cloud with LMDB storage, Cloudflare R2, and Railway.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Testing](#local-testing)
4. [Migration: Pickle → LMDB](#migration-pickle--lmdb)
5. [Cloud Setup](#cloud-setup)
6. [Deployment to Railway](#deployment-to-railway)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Guide Does

Transforms your local pickle-based audio fingerprinting system into a production-ready cloud deployment with:

- ✅ **70% faster startup** (90s → <10s) via LMDB memory-mapping
- ✅ **60% smaller storage** (3.86 GB → ~1.5 GB) via compression
- ✅ **Zero disk space issues** - FFmpeg pipes to memory
- ✅ **Cloud-ready** - Databases stored in Cloudflare R2
- ✅ **Auto-scaling** - Fast startup enables efficient scaling

### Architecture

```
Local Development:
- Pickle databases (backward compatible)
- Old code still works

Production Cloud:
- LMDB databases in Cloudflare R2
- Railway hosting
- Automatic database download on startup
- In-memory audio processing
```

---

## Prerequisites

### Required Software

1. **Python 3.10+**
   ```bash
   python3 --version  # Should be 3.10 or higher
   ```

2. **Docker** (for local testing)
   ```bash
   docker --version
   ```

3. **Git** (for deployment)
   ```bash
   git --version
   ```

### Required Accounts (All Free Tier Available)

1. **Cloudflare R2** (object storage)
   - Free tier: 10 GB storage, no egress fees
   - Cost: ~$0.02/month for this project

2. **Railway** (hosting platform)
   - Free $5 credit
   - Cost after: ~$5-10/month

3. **GitHub** (for code and CI/CD)
   - Free for public/private repos

4. **Sentry** (optional, for error tracking)
   - Free tier: 5,000 errors/month

---

## Local Testing

### Step 1: Install New Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs the new dependencies:
- `lmdb` - Memory-mapped database
- `xxhash` - Fast hashing
- `zstandard` - Compression
- `boto3` - AWS S3/R2 client

### Step 2: Test LMDB Mode Locally

```bash
# Set environment variable to enable LMDB mode
export USE_LMDB=true

# For local testing, limit to 5 seasons (faster startup)
export MAX_SEASONS=5

# Start the server
uvicorn app.main:app --reload
```

You should see:
```
🔧 LMDB mode enabled
📦 Fingerprint databases not found locally
   Falling back to empty database
```

This is expected - you haven't migrated yet!

**Note**: The `MAX_SEASONS` environment variable (default: 5) limits how many seasons are loaded. This is useful for local testing to reduce startup time and memory usage. For production, remove this limit or set it to 20.

### Step 3: Test with Docker

```bash
cd ..  # Back to project root

# Build Docker image
docker-compose build

# Test run (pickle mode)
docker-compose up

# Test run (LMDB mode with 5 seasons for local testing)
USE_LMDB=true MAX_SEASONS=5 docker-compose up
```

**Note**: By default, `MAX_SEASONS=5` is set in `docker-compose.yml` for local testing. This limits the backend to load only seasons 1-5, making startup faster and using less memory. For production deployment, you can remove this limit or set it to 20.

---

## Migration: Pickle → LMDB

### Step 1: Backup Your Pickle Databases

```bash
cd backend

# Create backup
mkdir -p backups
cp -r data/*.pkl backups/
```

### Step 2: Run Migration Script

```bash
# Migrate all seasons
python scripts/migrate_pickle_to_lmdb.py data data/fingerprints/v1
```

This will:
1. Load each pickle file
2. Convert to LMDB format
3. Compress with zstd
4. Save to `data/fingerprints/v1/season_XX.lmdb/`

Expected output:
```
============================================================
Migrating Pickle Databases to LMDB
============================================================

Migrating Season 01...
  Loaded 122,418 unique hashes
  Loaded 13 episodes
  Converting hashes...
  ✓ Migration complete!
    Pickle size: 211.8 MB
    LMDB size: 89.2 MB
    Reduction: 57.9%

...

Migration Summary
============================================================
Seasons migrated: 20
Total pickle size: 3.86 GB
Total LMDB size: 1.52 GB
Total reduction: 60.6%
✓ Migration complete!
```

### Step 3: Validate Migration

```bash
# Validate all seasons
python scripts/validate_migration.py data data/fingerprints/v1
```

Expected output:
```
Validating Season 01...
  Hash counts:
    Pickle: 122,418
    LMDB: 122,418
  Validating 1,000 random hashes...
  ✅ Validation PASSED!

...

Validation Summary
============================================================
Passed: 20
Failed: 0
✓ All validations passed!
```

### Step 4: Test LMDB Locally

```bash
# Start server in LMDB mode
export USE_LMDB=true
export DATA_DIR=./data

uvicorn app.main:app --reload
```

You should see:
```
🔧 Initializing LMDB audio fingerprint matcher...
📂 Opening LMDB databases...
   ✓ Opened season 01
   ✓ Opened season 02
   ...
   ✓ Opened 20 season databases
✓ LMDB initialization complete
```

### Step 5: Test a Match

Upload a test clip to `http://localhost:8000/docs` and try the `/api/v1/identify` endpoint.

---

## Cloud Setup

### 1. Create Cloudflare R2 Bucket

#### Sign Up

1. Go to https://dash.cloudflare.com/sign-up
2. Create account (free)
3. Verify email

#### Create R2 Bucket

1. In Cloudflare dashboard, click **R2** in sidebar
2. Click **Create bucket**
3. Name: `citywok-audio-db`
4. Location: **Automatic** (or choose closest region)
5. Click **Create bucket**

#### Get R2 Credentials

1. In R2 dashboard, click **Manage R2 API Tokens**
2. Click **Create API Token**
3. Token name: `citywok-production`
4. Permissions: **Object Read & Write**
5. Apply to buckets: **Specific buckets** → Select `citywok-audio-db`
6. Click **Create API Token**

**SAVE THESE - You'll need them!**
```
Access Key ID: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Secret Access Key: yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
Endpoint URL: https://<account-id>.r2.cloudflarestorage.com
```

### 2. Upload Databases to R2

#### Set Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your_r2_access_key"
export AWS_SECRET_ACCESS_KEY="your_r2_secret_key"
export AWS_ENDPOINT_URL="https://<account>.r2.cloudflarestorage.com"
export S3_BUCKET="citywok-audio-db"
```

#### Upload All Seasons

```bash
cd backend

# Upload all LMDB databases to R2
python offline_ingest.py --upload data/fingerprints/v1
```

Expected output:
```
============================================================
Uploading Databases to S3/R2
============================================================

Found 20 season databases

📦 Compressing season 01...
   Compressed to 42.3 MB
   Creating checksum...
   SHA256: abc123...
   Uploading to s3://citywok-audio-db/fingerprints/v1/season_01.lmdb.tar.gz...
   ✓ Checksum uploaded
   ✓ Season 01 uploaded successfully

...

Upload Summary
============================================================
Successful: 20
Failed: 0
✓ All uploads complete!
```

#### Verify Upload

```bash
# List files in R2
aws s3 ls s3://citywok-audio-db/fingerprints/v1/ --endpoint-url=$AWS_ENDPOINT_URL
```

You should see:
```
season_01.lmdb.tar.gz
season_02.lmdb.tar.gz
...
season_20.lmdb.tar.gz
```

### 3. Create Railway Account

#### Sign Up

1. Go to https://railway.app
2. Click **Start a New Project**
3. Sign in with GitHub (recommended)
4. Verify email

#### Create New Project

1. Click **New Project**
2. Select **Deploy from GitHub repo**
3. Choose your `CityWok` repository
4. Railway will detect the Dockerfile automatically

### 4. Configure Railway Project

#### Set Environment Variables

In Railway dashboard:

1. Click your project
2. Click **Variables** tab
3. Add these variables:

```
# LMDB Mode
USE_LMDB=true
DB_VERSION=v1

# R2 Credentials
S3_BUCKET=citywok-audio-db
S3_PREFIX=fingerprints
AWS_ACCESS_KEY_ID=<your_r2_access_key>
AWS_SECRET_ACCESS_KEY=<your_r2_secret_key>
AWS_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
AWS_REGION=auto

# Data Directory
DATA_DIR=/app/data
```

#### Configure Service

1. Click **Settings** tab
2. **Root Directory**: `backend`
3. **Build Command**: (leave default)
4. **Start Command**: (uses Dockerfile CMD)
5. **Port**: `8000` (Railway will auto-detect)

#### Add Persistent Volume (Optional but Recommended)

This caches downloaded databases between deployments:

1. In Railway dashboard, click **Variables** tab
2. Scroll to **Volumes**
3. Click **+ New Volume**
4. Mount path: `/app/data/fingerprints`
5. Size: 2 GB (minimum for all databases)

---

## Deployment to Railway

### Method 1: Automatic Deployment (Recommended)

Railway automatically deploys when you push to `main`:

```bash
git add .
git commit -m "Deploy LMDB production version"
git push origin main
```

Railway will:
1. Detect the push
2. Build Docker image
3. Download databases from R2 (on first run)
4. Start the service
5. Provide you with a URL

### Method 2: Manual Deployment via CLI

#### Install Railway CLI

```bash
# macOS
brew install railway

# Linux/Windows
npm install -g @railway/cli
```

#### Login and Deploy

```bash
# Login
railway login

# Link to your project
cd /path/to/CityWok
railway link

# Deploy
railway up
```

### Method 3: GitHub Actions (Automated)

Already configured! Just push to `main`:

```bash
git push origin main
```

The workflow (`.github/workflows/deploy-railway.yml`) will deploy automatically.

---

## Monitor Deployment

### Check Railway Logs

In Railway dashboard:
1. Click **Deployments**
2. Click latest deployment
3. View logs

Look for:
```
========================================
CityWok Audio Fingerprinting API
========================================

Configuration:
  USE_LMDB: true
  DATA_DIR: /app/data

🔧 LMDB mode enabled
📦 Downloading fingerprint databases from S3
   Bucket: s3://citywok-audio-db/fingerprints/v1/

  Downloading season_01.lmdb.tar.gz...
    Downloaded 42.3 MB
    Verifying checksum...
    ✓ Checksum verified
    Extracting...
    ✓ Season 01 ready

...

✓ Downloaded 20 season databases

========================================
Starting API server...
========================================

🔧 Initializing LMDB audio fingerprint matcher...
   ✓ Opened season 01
   ...
   ✓ Opened 20 season databases
✓ LMDB initialization complete

✓ Startup complete!
```

### Test Your Deployment

Railway gives you a URL like: `https://citywok-production.railway.app`

Test it:
```bash
curl https://your-app.railway.app/api/v1/test
```

Expected response:
```json
{
  "status": "ok",
  "message": "CityWok API is running (Audio-Only)",
  "features": {
    "audio": true,
    "url_download": true
  }
}
```

---

## Monitoring & Maintenance

### Set Up Sentry (Optional)

#### Create Sentry Account

1. Go to https://sentry.io/signup
2. Sign up (free tier)
3. Create new project: **Python → FastAPI**
4. Copy your DSN

#### Add Sentry to Your App

1. Add to `requirements.txt`:
   ```
   sentry-sdk[fastapi]==1.40.0
   ```

2. Update `app/main.py`:
   ```python
   import sentry_sdk

   sentry_sdk.init(
       dsn="your_sentry_dsn_here",
       environment="production"
   )
   ```

3. Add DSN to Railway environment variables:
   ```
   SENTRY_DSN=your_sentry_dsn_here
   ```

### Monitor Railway Metrics

Railway provides built-in metrics:
- CPU usage
- Memory usage
- Network traffic
- Request logs

Access in Railway dashboard → **Metrics** tab.

### Health Checks

Railway automatically monitors `/` (health check endpoint).

To manually check:
```bash
curl https://your-app.railway.app/api/v1/test
```

---

## Updating Databases

### Add New Episodes

#### Method 1: Local Ingestion + Upload

```bash
# Ingest new season locally
python backend/offline_ingest.py "/path/to/Season 21" 21

# Upload to R2
python backend/offline_ingest.py --upload backend/data/fingerprints/v1
```

#### Method 2: Direct API Ingestion (Smaller Updates)

Use the `/api/v1/ingest` endpoint:

```bash
curl -X POST "https://your-app.railway.app/api/v1/ingest" \
  -F "episode_id=S21E01" \
  -F "file=@episode.mp4"
```

### Trigger Railway Redeploy

After uploading new databases to R2:

```bash
# Force Railway to restart and download new databases
railway restart
```

Or in Railway dashboard:
1. Click **Deployments**
2. Click **⋯** (three dots)
3. Click **Restart**

---

## Troubleshooting

### Issue: "Database download failed"

**Cause**: R2 credentials not set correctly

**Fix**:
1. Check Railway environment variables
2. Verify `AWS_ENDPOINT_URL` is correct for your R2 account
3. Test credentials locally:
   ```bash
   aws s3 ls s3://citywok-audio-db --endpoint-url=$AWS_ENDPOINT_URL
   ```

### Issue: "Memory limit exceeded"

**Cause**: Railway free tier has 512 MB RAM limit

**Fix**:
1. Upgrade Railway plan to 1 GB RAM ($5/month)
2. Or reduce number of loaded seasons (modify code to load on-demand)

### Issue: "Startup timeout"

**Cause**: Downloading 1.5 GB databases takes time

**Fix**:
1. Use persistent volume (caches databases)
2. Increase Railway timeout (in settings)

### Issue: "Matches not working after migration"

**Cause**: Migration validation failed

**Fix**:
1. Run validation script locally:
   ```bash
   python scripts/validate_migration.py data data/fingerprints/v1
   ```
2. Check for errors
3. Re-run migration if needed

### Issue: "Want to roll back to pickle mode"

**Fix**:
```bash
# In Railway, set environment variable:
USE_LMDB=false

# Or locally:
export USE_LMDB=false
uvicorn app.main:app
```

---

## Cost Summary

### Monthly Costs (Estimate)

| Service | Cost | Notes |
|---------|------|-------|
| **Cloudflare R2** | $0.02/month | 1.5 GB storage + zero egress |
| **Railway** | $5-10/month | 512 MB RAM, 0.5 vCPU |
| **Sentry** | $0/month | Free tier (5k errors) |
| **GitHub Actions** | $0/month | Free for public repos |
| **Total** | **$5-10/month** | 💰 Very affordable! |

### Comparison to AWS

| AWS Service | Cost |
|-------------|------|
| EC2 t3.small | $15/month |
| S3 storage + egress | $4/month |
| CloudWatch | $3/month |
| **AWS Total** | **$22/month** |

**Savings: ~$12/month (54% cheaper)**

---

## Quick Reference

### Essential Commands

```bash
# Local Testing
export USE_LMDB=true
uvicorn app.main:app --reload

# Migration
python scripts/migrate_pickle_to_lmdb.py data data/fingerprints/v1
python scripts/validate_migration.py data data/fingerprints/v1

# Upload to R2
python offline_ingest.py --upload data/fingerprints/v1

# Deploy to Railway
git push origin main

# Check Railway logs
railway logs

# Restart Railway
railway restart
```

### Environment Variables

| Variable | Value | Used For |
|----------|-------|----------|
| `USE_LMDB` | `true` | Enable LMDB mode |
| `AWS_ACCESS_KEY_ID` | R2 access key | R2 authentication |
| `AWS_SECRET_ACCESS_KEY` | R2 secret key | R2 authentication |
| `AWS_ENDPOINT_URL` | `https://<account>.r2.cloudflarestorage.com` | R2 endpoint |
| `S3_BUCKET` | `citywok-audio-db` | R2 bucket name |

### Important URLs

- **Railway Dashboard**: https://railway.app/dashboard
- **Cloudflare R2**: https://dash.cloudflare.com → R2
- **GitHub Repo**: https://github.com/yourusername/CityWok
- **Sentry**: https://sentry.io/organizations/yourorg

---

## Next Steps

1. ✅ Complete local testing
2. ✅ Run migration and validation
3. ✅ Create Cloudflare R2 account
4. ✅ Upload databases to R2
5. ✅ Create Railway account
6. ✅ Deploy to Railway
7. ✅ Test production deployment
8. ✅ Set up monitoring (optional)

**Questions?** Check the troubleshooting section or open a GitHub issue!

---

## Success Checklist

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Migration completed successfully
- [ ] Validation passed (all seasons)
- [ ] Cloudflare R2 bucket created
- [ ] R2 credentials obtained
- [ ] Databases uploaded to R2
- [ ] Railway account created
- [ ] Environment variables configured in Railway
- [ ] Deployment successful
- [ ] API responding at Railway URL
- [ ] Test match works in production
- [ ] Monitoring set up (optional)

**🎉 Congratulations! Your CityWok backend is now running in the cloud!**

---

*Total setup time: 1-2 hours*
*Monthly cost: ~$5-10*
*Performance improvement: 70% faster startup, 60% smaller storage*

