# CityWok - Audio Fingerprinting Episode Identifier

A production-ready, full-stack web application that identifies TV show episodes from short audio/video clips using advanced audio fingerprinting technology. Built with modern Python and React, featuring enterprise-grade performance optimizations and security.

## 🎯 Project Overview

CityWok is a sophisticated audio recognition system that can identify specific episodes and timestamps from short clips (as brief as 3-5 seconds), even when audio is distorted, compressed, or filtered. The system processes audio from TikTok videos, YouTube clips, Instagram reels, or direct file uploads and matches them against a comprehensive database of 20 seasons.

## ✨ Key Technical Achievements

### Advanced Audio Processing
- **Custom Audio Fingerprinting Algorithm**: Implemented a spectrogram-based peak detection system using librosa and NumPy for robust audio analysis
- **Optimized Matching Engine**: Multi-stage incremental matching with early exit optimizations for sub-second response times
- **Parallel Processing**: ThreadPoolExecutor-based concurrent season matching for optimal performance
- **Memory-Efficient Architecture**: Handles 6.5GB+ database with intelligent lazy loading and memory management

### Full-Stack Development
- **Backend**: FastAPI REST API with async/await patterns, Server-Sent Events (SSE) for real-time progress updates
- **Frontend**: Modern React application with responsive design, drag-and-drop file uploads, and streaming progress indicators
- **Cloud Infrastructure**: Containerized deployment on Railway with Cloudflare R2 storage integration

### Production-Ready Features
- **Security**: Web Application Firewall (WAF), security headers (CSP, HSTS), IP-based rate limiting, API key authentication
- **Scalability**: Supports 1000+ daily users with configurable rate limits (10 req/hr free tier, 100 req/hr authenticated)
- **Error Handling**: Comprehensive error handling with user-friendly messages and graceful degradation
- **Monitoring**: Sentry integration for production error tracking and performance monitoring

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI (async Python web framework)
- **Audio Processing**: librosa, NumPy, SciPy for signal processing and spectrogram analysis
- **Data Storage**: Pickle-based fingerprint databases stored in Cloudflare R2 (S3-compatible)
- **Video Processing**: yt-dlp for multi-platform video download and audio extraction
- **Deployment**: Docker containerization, Railway hosting with persistent volumes

### Frontend
- **Framework**: React 18 with Vite build tool
- **Styling**: Custom CSS with responsive design principles
- **Deployment**: Vercel with environment-based configuration
- **Analytics**: Vercel Analytics integration

### Infrastructure
- **Backend Hosting**: Railway (containerized deployment)
- **Storage**: Cloudflare R2 (object storage for database files)
- **Frontend Hosting**: Vercel (edge network deployment)
- **Monitoring**: Sentry for error tracking

## 🏗️ Architecture Highlights

### Audio Fingerprinting Pipeline
1. **Audio Extraction**: Downloads videos from URLs (TikTok, YouTube, Instagram) or processes uploaded files
2. **Preprocessing**: Converts to 22kHz mono, extracts spectrogram using Short-Time Fourier Transform (STFT)
3. **Peak Detection**: Identifies significant frequency peaks using maximum filter algorithms
4. **Hash Generation**: Creates unique fingerprints using MD5 hashing of peak constellations
5. **Database Matching**: Parallel search across 20 seasons using optimized numpy arrays
6. **Result Ranking**: Confidence scoring with multiple quality metrics (alignment, margin, sharpness)

### Performance Optimizations
- **Incremental Matching**: Processes audio in stages (3s, 5s, 8s, 10s) with early exit on confident matches
- **Multi-Window Sampling**: Extracts and matches multiple time windows for robust distorted clip handling
- **IDF Weighting**: Down-weights common audio patterns to improve match accuracy
- **Posting List Capping**: Limits per-hash matches to prevent noise from overpowering signals
- **Lazy Loading**: On-demand database loading to optimize memory usage

### Data Management
- **Database Format**: Efficient numpy array storage with structured dtypes for minimal memory footprint
- **Season-Based Organization**: Separate databases per season for parallel processing
- **R2 Integration**: Automated download scripts for cloud-stored databases
- **Volume Persistence**: Railway volumes for persistent storage across deployments

## 🔒 Security & Production Features

- **Rate Limiting**: IP-based throttling (10 requests/hour free, 100/hour with API key)
- **API Authentication**: Optional API key system for enhanced rate limits
- **WAF Protection**: Custom middleware for request validation and attack prevention
- **Security Headers**: Comprehensive HTTP security headers (CSP, HSTS, X-Frame-Options, etc.)
- **CORS Configuration**: Strict origin validation with environment-based configuration
- **Input Validation**: File type and size validation, URL pattern matching
- **Error Sanitization**: Generic error messages to prevent information leakage

## 📊 Performance Metrics

- **Database Size**: 6.5GB across 20 seasons
- **Memory Usage**: Optimized to handle large datasets with lazy loading capabilities
- **Response Time**: Sub-second matching for short clips with early exit optimizations
- **Scalability**: Designed to handle 1000+ concurrent users with rate limiting
- **Accuracy**: High-precision matching even with audio distortion and compression

## 🚀 Deployment

The application is fully containerized and deployed on modern cloud platforms:

- **Backend**: Railway with Docker, persistent volumes, and environment-based configuration
- **Frontend**: Vercel with automatic deployments from Git
- **Storage**: Cloudflare R2 for cost-effective object storage

## 🎓 Technical Skills Demonstrated

- **Full-Stack Development**: End-to-end application development from backend API to frontend UI
- **Audio Signal Processing**: Deep understanding of spectrograms, STFT, peak detection, and audio fingerprinting
- **Algorithm Design**: Custom matching algorithms with multiple optimization strategies
- **Performance Optimization**: Memory management, parallel processing, incremental algorithms
- **Cloud Architecture**: Multi-service deployment, object storage integration, containerization
- **Security Implementation**: Production-grade security measures and best practices
- **Modern Python**: Async/await, type hints, modern dependency management
- **Modern JavaScript**: React hooks, Server-Sent Events, responsive design patterns

## 📝 License

MIT License - Open source and available for learning and modification.

---

*Built with attention to performance, security, and user experience. A production-ready example of modern full-stack development with advanced signal processing.*
