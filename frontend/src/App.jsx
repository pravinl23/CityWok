import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import cartmanImage from './assets/cartman.png'
import backgroundImage from './assets/background.jpg'
import backgroundMobile from './assets/background-mobile.jpg'

// Get API URL from environment variable or use localhost for development
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [url, setUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [currentGuessIndex, setCurrentGuessIndex] = useState(0)
  const [progress, setProgress] = useState(0)
  const fileInputRef = useRef(null)

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

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

  const validateUrl = (urlString) => {
    // Check if URL is from supported platforms: TikTok, Instagram, or YouTube
    const supportedPlatforms = [
      /^https?:\/\/(www\.)?(tiktok\.com|vm\.tiktok\.com)/i,  // TikTok
      /^https?:\/\/(www\.)?(instagram\.com|instagr\.am)/i,   // Instagram
      /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)/i,       // YouTube
    ];

    return supportedPlatforms.some(pattern => pattern.test(urlString));
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
    setStatusMessage('')
    setCurrentGuessIndex(0)
    setProgress(0)
    
    const formData = new FormData()
    formData.append('file', file)
    formData.append('stream', 'true')
    
    try {
      // Use fetch for Server-Sent Events
      const response = await fetch(`${API_URL}/api/v1/identify`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header, browser will set it with boundary
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              // Handle progress updates
              if (data.status) {
                const statusToProgress = {
                  'uploading': 10,
                  'downloading': 20,
                  'extracting': 40,
                  'converting': 50,
                  'fingerprinting': 70,
                  'searching': 85,
                  'matching': 95,
                  'complete': 100
                }
                setProgress(statusToProgress[data.status] || 0)
                setStatusMessage(data.message || getStatusMessage(data.status))
              }
              
              // Handle final result
              if (data.match_found !== undefined) {
                setResult(data)
                setStatusMessage('')
              }
              
              // Handle errors
              if (data.error) {
                setError(data.error)
                setStatusMessage('')
              }
            } catch (e) {
              // Ignore parsing errors
            }
          }
        }
      }
    } catch (err) {
      // Enhanced error handling
      let errorMessage = "An unexpected error occurred. Please try again.";
      
      if (err.response?.status === 429) {
        errorMessage = "Rate limit exceeded. You can make 10 requests per hour. Please try again later.";
      } else if (err.response?.status === 400) {
        errorMessage = err.response?.data?.detail || "Invalid request. Please check your file.";
      } else if (err.response?.status === 502 || err.response?.status === 503) {
        errorMessage = "Server is temporarily unavailable. Please try again in a few moments.";
      } else if (err.name === 'AbortError' || err.message.includes('timeout')) {
        errorMessage = "Request timed out. Your file might be too long (max 2 minutes recommended).";
      } else if (!navigator.onLine) {
        errorMessage = "No internet connection. Please check your network and try again.";
      } else if (err.message.includes('HTTP error')) {
        errorMessage = "Server error occurred. Please try again later.";
      }
      
      setError(errorMessage);
      setStatusMessage('')
    } finally {
      setLoading(false)
    }
  }
  
  const getStatusMessage = (status) => {
    const messages = {
      'uploading': 'Processing file...',
      'downloading': 'Downloading video...',
      'extracting': 'Extracting audio...',
      'converting': 'Preparing audio...',
      'fingerprinting': 'Analyzing audio...',
      'searching': 'Searching for matches...',
      'matching': 'Verifying match...',
      'complete': 'Complete!'
    }
    return messages[status] || 'Processing...'
  }

  const handleUrlSubmit = async () => {
    if (!url.trim()) {
      setError("Please enter a URL")
      return
    }

    // Validate URL is from supported platform
    if (!validateUrl(url.trim())) {
      setError("Please enter a valid TikTok, Instagram, or YouTube URL")
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setStatusMessage('')
    setCurrentGuessIndex(0)
    setProgress(0)
    
    const formData = new FormData()
    formData.append('url', url.trim())
    formData.append('stream', 'true')
    
    try {
      // Use fetch for Server-Sent Events
      const response = await fetch(`${API_URL}/api/v1/identify`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              // Handle progress updates
              if (data.status) {
                const statusToProgress = {
                  'uploading': 10,
                  'downloading': 20,
                  'extracting': 40,
                  'converting': 50,
                  'fingerprinting': 70,
                  'searching': 85,
                  'matching': 95,
                  'complete': 100
                }
                setProgress(statusToProgress[data.status] || 0)
                setStatusMessage(data.message || getStatusMessage(data.status))
              }
              
              // Handle final result
              if (data.match_found !== undefined) {
                setResult(data)
                setStatusMessage('')
              }
              
              // Handle errors
              if (data.error) {
                setError(data.error)
                setStatusMessage('')
              }
            } catch (e) {
              // Ignore parsing errors
            }
          }
        }
      }
    } catch (err) {
      // Enhanced error handling
      let errorMessage = "An unexpected error occurred. Please try again.";
      
      if (err.response?.status === 429) {
        errorMessage = "Rate limit exceeded. You can make 10 requests per hour. Please try again later.";
      } else if (err.response?.status === 400) {
        errorMessage = err.response?.data?.detail || "Invalid request. Please check your URL.";
      } else if (err.response?.status === 502 || err.response?.status === 503) {
        errorMessage = "Server is temporarily unavailable. Please try again in a few moments.";
      } else if (err.name === 'AbortError' || err.message.includes('timeout')) {
        errorMessage = "Request timed out. The URL might be unavailable or the server is busy.";
      } else if (!navigator.onLine) {
        errorMessage = "No internet connection. Please check your network and try again.";
      } else if (err.message.includes('HTTP error')) {
        errorMessage = "Server error occurred. Please try again later.";
      }
      
      setError(errorMessage);
      setStatusMessage('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="background" style={{ backgroundImage: `url(${isMobile ? backgroundMobile : backgroundImage})` }}>
      </div>

      <div className="banner-container">
        <div className="banner banner-single">FIND THE EPISODE, M'KAY?</div>
      </div>

      <div className="content">
        <div className="left-side">
          <div className="cartman">
            <img src={cartmanImage} alt="Cartman" className="cartman-image" />
          </div>
        </div>

        <div className="right-side">
          <div className={`upload-card ${isDragging ? 'dragging' : ''}`}>
            <div 
              className="input-section"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                const droppedFile = e.dataTransfer.files[0];
                if (droppedFile) {
                  setUrl(''); // Clear URL if file is dropped
                  handleFileChange(droppedFile);
                }
              }}
            >
              <input
                type="text"
                placeholder="Paste TikTok, Instagram, or YouTube URL..."
                value={file ? file.name : url}
                onChange={(e) => {
                  setUrl(e.target.value);
                  setFile(null); // Clear file when typing URL
                }}
                className="combined-input"
                readOnly={!!file}
              />
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm,.mp3,.wav,.m4a,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,audio/mpeg,audio/wav"
                onChange={handleFileInputChange}
                style={{ display: 'none' }}
              />
              <button 
                className="find-episode-btn"
                onClick={() => {
                  if (file) {
                    handleIdentify();
                  } else if (url.trim()) {
                    handleUrlSubmit();
                  }
                }}
                disabled={(!file && !url.trim()) || loading}
              >
                {loading ? 'PROCESSING...' : 'FIND EPISODE'}
              </button>
            </div>
            
            {loading && (
              <div className="loading-container">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="status-message">{statusMessage}</p>
              </div>
            )}

            <button 
              className="upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
            >
              UPLOAD FILE
            </button>

            {error && <div className="error-message">{error}</div>}

            {result && (
              <div className="result-card">
                {result.unsure && result.candidates && result.candidates.length > 0 ? (
                  <>
                    <h3 className="result-title">
                      {currentGuessIndex === 0 ? "I'm not quite sure, but this is my best guess" : `Guess ${currentGuessIndex + 1}`}
                    </h3>
                    <div className="primary-result">
                      <p className="result-episode">Episode: {result.candidates[currentGuessIndex].episode_id}</p>
                      <div className="result-details">
                        <small>Confidence: {result.candidates[currentGuessIndex].confidence}%</small>
                        <br/>
                        <small>Aligned: {result.candidates[currentGuessIndex].aligned_matches}</small>
                      </div>
                    </div>
                    
                    {currentGuessIndex < result.candidates.length - 1 ? (
                      <button 
                        className="wrong-guess-btn"
                        onClick={() => setCurrentGuessIndex(currentGuessIndex + 1)}
                      >
                        This is wrong
                      </button>
                    ) : (
                      <div className="no-more-guesses">
                        <p>I have no more guesses for this clip. Sorry!</p>
                      </div>
                    )}
                  </>
                ) : result.match_found ? (
                  <>
                    <h3 className="result-title">Match Found!</h3>
                    <div className="primary-result">
                      <p className="result-episode">Episode: {result.episode}</p>
                      <p className="result-timestamp">Time: {result.timestamp}</p>
                      <div className="result-details">
                        <small>Confidence: {result.confidence}%</small>
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
                    <p>{result.message || "No matches found in database"}</p>
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
