import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'
import cartmanImage from './assets/cartman.png'
import backgroundImage from './assets/background.jpg'

// Get API URL from environment variable or use localhost for development
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [url, setUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const validateFile = (selectedFile) => {
    // Check file size (100MB limit)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (selectedFile.size > maxSize) {
      setError(`File too large! Maximum size is 100MB. Your file is ${(selectedFile.size / (1024 * 1024)).toFixed(1)}MB.`);
      return false;
    }

    // Check file type
    const validTypes = [
      'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm',
      'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/x-m4a'
    ];
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase();
    const validExtensions = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'm4a'];

    if (!validTypes.includes(selectedFile.type) && !validExtensions.includes(fileExtension)) {
      setError(`Invalid file type! Please upload MP4, MOV, AVI, MKV, MP3, WAV, or M4A files.`);
      return false;
    }

    return true;
  }

  const handleFileChange = (selectedFile) => {
    if (selectedFile) {
      setError(null);
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
        setResult(null);
      } else {
        setFile(null);
      }
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      handleFileChange(droppedFile)
    }
  }

  const handleFileInputChange = (e) => {
    if (e.target.files[0]) {
      handleFileChange(e.target.files[0])
    }
  }

  const handleIdentify = async () => {
    if (!file) return
    
    setLoading(true)
    setError(null)
    setResult(null)
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await axios.post(`${API_URL}/api/v1/identify`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 300000
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)

      // Handle different error types
      if (err.code === 'ECONNABORTED' || err.message.includes('timeout')) {
        setError("Request timed out. The file might be too large or the server is busy. Please try again.");
      } else if (err.response) {
        // Server responded with error
        const status = err.response.status;
        const detail = err.response.data?.detail || err.response.data?.message;

        if (status === 400) {
          setError(detail || "Invalid file or request. Please check your file and try again.");
        } else if (status === 413) {
          setError("File too large! Maximum size is 100MB.");
        } else if (status === 500) {
          setError("Server error occurred. Please try again later.");
        } else if (status === 502 || status === 503) {
          setError("Server is temporarily unavailable. Please try again in a moment.");
        } else {
          setError(detail || `Error ${status}: An error occurred during identification.`);
        }
      } else if (err.request) {
        // Request made but no response
        setError("Cannot connect to server. Please check your internet connection or try again later.");
      } else {
        setError("An unexpected error occurred. Please try again.");
      }
    } finally {
      setLoading(false)
    }
  }

  const handleUrlSubmit = async () => {
    if (!url.trim()) {
      setError("Please enter a URL")
      return
    }
    
    setLoading(true)
    setError(null)
    setResult(null)
    
    const formData = new FormData()
    formData.append('url', url.trim())
    
    try {
      const response = await axios.post(`${API_URL}/api/v1/identify`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 300000
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)

      // Handle different error types for URL submission
      if (err.code === 'ECONNABORTED' || err.message.includes('timeout')) {
        setError("Request timed out. The URL might be unavailable or the server is busy. Please try again.");
      } else if (err.response) {
        const status = err.response.status;
        const detail = err.response.data?.detail || err.response.data?.message;

        if (status === 400) {
          setError(detail || "Invalid URL. Please check the URL and try again.");
        } else if (status === 408) {
          setError("Download timeout. The video took too long to download. Try a shorter clip.");
        } else if (status === 500) {
          setError("Server error occurred. Please try again later.");
        } else if (status === 502 || status === 503) {
          setError("Server is temporarily unavailable. Please try again in a moment.");
        } else {
          setError(detail || `Error ${status}: Could not process URL.`);
        }
      } else if (err.request) {
        setError("Cannot connect to server. Please check your internet connection or try again later.");
      } else {
        setError("An unexpected error occurred. Please try again.");
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="background" style={{ backgroundImage: `url(${backgroundImage})` }}>
      </div>

      <div className="content">
        <div className="left-side">
          <div className="banners">
            <div className="banner banner-top">FIND THE</div>
            <div className="banner banner-bottom">EPISODE, M'KAY?</div>
          </div>
          <div className="cartman">
            <img src={cartmanImage} alt="Cartman" className="cartman-image" />
          </div>
        </div>

        <div className="right-side">
          <div className={`upload-card ${isDragging ? 'dragging' : ''}`}>
            <div 
              className="drop-zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-icon">↑</div>
              <p className="upload-text">Drag & Drop Media</p>
              <p className="upload-formats">(MP4, MP3, WAV, M4A)</p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm,.mp3,.wav,.m4a,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,audio/mpeg,audio/wav"
                onChange={handleFileInputChange}
                style={{ display: 'none' }}
              />
              {file && (
                <p className="file-name">{file.name}</p>
              )}
            </div>


            <button 
              className="upload-btn"
              onClick={handleIdentify}
              disabled={!file || loading}
            >
              {loading ? 'PROCESSING...' : 'UPLOAD FILE'}
            </button>

            <p className="or-text">OR PASTE URL</p>
            
            <div className="url-section">
              <input
                type="text"
                placeholder="https://tiktok.com/..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="url-input"
              />
              <button 
                className="find-btn"
                onClick={handleUrlSubmit}
                disabled={!url.trim()}
              >
                FIND IT
        </button>
            </div>

            {error && <div className="error-message">{error}</div>}

            {result && (
              <div className="result-card">
                {result.match_found ? (
                  <>
                    <h3 className="result-title">Match Found!</h3>
                    <div className="primary-result">
                      <p className="result-episode">Episode: {result.episode}</p>
                      <p className="result-timestamp">Time: {result.timestamp}</p>
                      <div className="result-details">
                        <small>Confidence: {result.confidence}</small>
                        {result.aligned_matches && (
                          <>
                            <br/>
                            <small>Matched: {result.aligned_matches} fingerprints</small>
                          </>
                        )}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="no-match">
                    <h3>No Match Found</h3>
                    <p>{result.message}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
