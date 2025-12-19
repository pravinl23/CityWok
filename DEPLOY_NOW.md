# 🚀 Quick Deployment Guide - CityWok

## Option 1: Deploy with Pickle Files (Fastest - Use This First!)

Since you already have all 20 seasons in pickle format and the system is working, you can deploy immediately.

### Step 1: Prepare Your Code

```bash
cd /Users/pravinlohani/Projects/CityWok

# Make sure everything is committed
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Create Railway Account

1. Go to https://railway.app
2. Sign up with GitHub (recommended)
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your `CityWok` repository

### Step 3: Configure Railway

#### A. Set Root Directory
1. In Railway dashboard, click your project
2. Click **Settings** tab
3. Set **Root Directory**: `backend`

#### B. Set Environment Variables
1. Click **Variables** tab
2. Add these variables:

```
# Use pickle mode (your existing databases)
USE_LMDB=false

# Data directory
DATA_DIR=/app/data

# Load all 20 seasons (remove MAX_SEASONS to load all)
# MAX_SEASONS=20  # Optional: uncomment to explicitly set

# Python settings
PYTHONUNBUFFERED=1
```

#### C. Add Persistent Volume (Important!)
This stores your pickle database files:

1. In **Variables** tab, scroll to **Volumes**
2. Click **+ New Volume**
3. Mount path: `/app/data`
4. Size: **5 GB** (enough for all pickle files)

### Step 4: Upload Databases to Volume

After Railway creates the volume, you need to upload your pickle files:

#### Option A: Via Railway CLI (Recommended)

```bash
# Install Railway CLI
brew install railway  # macOS
# OR
npm install -g @railway/cli  # Linux/Windows

# Login
railway login

# Link to your project
cd /Users/pravinlohani/Projects/CityWok
railway link

# Upload databases
cd backend/data
railway run --service api -- sh -c "mkdir -p /app/data && echo 'Volume ready'"
```

Then use Railway's file upload feature or SSH into the container to copy files.

#### Option B: Via Docker Volume Mount (Local Testing First)

For local testing, you can mount your data directory:

```bash
# Test locally with Docker
cd /Users/pravinlohani/Projects/CityWok
docker-compose up
```

### Step 5: Deploy

Railway will automatically deploy when you push to `main`. Or manually:

```bash
# In Railway dashboard, click "Deploy" button
# OR via CLI:
railway up
```

### Step 6: Test Your Deployment

Railway will give you a URL like: `https://citywok-production.railway.app`

Test it:
```bash
curl https://your-app.railway.app/api/v1/test
```

Test identification:
```bash
curl -X POST "https://your-app.railway.app/api/v1/identify" \
  -F "url=https://www.tiktok.com/@tik_tok_cliped/video/7209768566252490026?is_from_webapp=1&sender_device=pc&web_id=7493733278417389111"
```

---

## Option 2: Deploy with LMDB (Optimized - For Later)

This is more optimized (60% smaller, 70% faster startup) but requires migration first.

### Quick Migration Steps:

```bash
cd /Users/pravinlohani/Projects/CityWok/backend

# 1. Migrate pickle → LMDB
python scripts/migrate_pickle_to_lmdb.py data data/fingerprints/v1

# 2. Validate migration
python scripts/validate_migration.py data data/fingerprints/v1

# 3. Upload to Cloudflare R2 (see DEPLOYMENT_GUIDE.md for details)
# 4. Set USE_LMDB=true in Railway
```

See `DEPLOYMENT_GUIDE.md` for full LMDB deployment instructions.

---

## Current Status ✅

Your system is **ready to deploy** with pickle files:
- ✅ All 20 seasons ingested
- ✅ Backend tested and working
- ✅ TikTok URL identification working
- ✅ MAX_SEASONS support added

## Recommended: Start with Option 1

Deploy with pickle files first to get it live quickly, then optimize to LMDB later if needed.

---

## Troubleshooting

### Issue: "No databases found"
- Make sure persistent volume is mounted at `/app/data`
- Upload your pickle files to the volume
- Check Railway logs: `railway logs`

### Issue: "Out of memory"
- Railway free tier has 512 MB RAM
- Consider upgrading to 1 GB plan ($5/month)
- Or reduce MAX_SEASONS for testing

### Issue: "Slow startup"
- First startup loads all databases (takes ~30-60 seconds)
- Subsequent restarts are faster
- Consider LMDB migration for faster startup

---

## Quick Commands Reference

```bash
# Check Railway logs
railway logs

# Restart service
railway restart

# View environment variables
railway variables

# SSH into container (if available)
railway shell
```

---

## Next Steps After Deployment

1. ✅ Test the API endpoint
2. ✅ Test TikTok URL identification
3. ✅ Monitor Railway metrics
4. ✅ Set up custom domain (optional)
5. ✅ Consider LMDB migration for optimization

**You're ready to deploy! 🚀**

