# CityWok - Audio Fingerprinting API

Audio fingerprinting service for identifying episodes from short clips.

## Features

- ✅ Audio fingerprinting with high accuracy
- ✅ Support for TikTok, YouTube, Instagram URLs
- ✅ File upload support
- ✅ API key authentication (optional)
- ✅ Rate limiting (10 req/hr free, 100 req/hr with API key)
- ✅ WAF protection
- ✅ Security headers

## Quick Start

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings
```

### Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Deploy to Railway

1. Push to GitHub
2. Connect repo to Railway
3. Add environment variables:
   ```bash
   LAZY_LOAD_PICKLE=true
   MATCH_WORKERS=8
   ALLOWED_ORIGINS=*
   ```
4. Deploy!

## API Usage

### Health Check

```bash
curl https://your-app.railway.app/api/v1/health
```

### Identify Episode

```bash
# From URL
curl -X POST https://your-app.railway.app/api/v1/identify \
  -F "url=https://tiktok.com/@user/video/123"

# From file
curl -X POST https://your-app.railway.app/api/v1/identify \
  -F "file=@video.mp4"

# With API key (100 req/hr)
curl -X POST https://your-app.railway.app/api/v1/identify \
  -H "X-API-Key: sk_live_..." \
  -F "url=https://tiktok.com/@user/video/123"
```

## Security

- IP-based rate limiting
- API key authentication
- WAF protection
- Security headers (CSP, HSTS, etc.)
- CORS hardening

See `SECURITY.md` for details.

## License

MIT
