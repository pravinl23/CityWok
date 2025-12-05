import { useState, useRef } from 'react'
import axios from 'axios'
import './App.css'
import cartmanImage from './assets/cartman.png'
import backgroundImage from './assets/background.jpg'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [url, setUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileChange = (selectedFile) => {
    if (selectedFile) {
      setFile(selectedFile)
      setResult(null)
      setError(null)
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
      const response = await axios.post('http://localhost:8000/api/v1/identify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 300000
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || "An error occurred during identification.")
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
      const response = await axios.post('http://localhost:8000/api/v1/identify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 300000
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || "An error occurred during identification.")
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
                accept="video/*,audio/*"
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
