import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('identify') // 'identify' or 'ingest'
  const [episodeId, setEpisodeId] = useState('')

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
    setResult(null)
    setError(null)
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
        timeout: 300000  // 5 minute timeout for processing
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || "An error occurred during identification.")
    } finally {
      setLoading(false)
    }
  }

  const handleIngest = async () => {
    if (!file || !episodeId) {
      setError("Please select a file and enter an Episode ID")
      return
    }

    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      // Append episode_id to query params
      const response = await axios.post(`http://localhost:8000/api/v1/ingest?episode_id=${episodeId}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResult({ message: response.data.message })
    } catch (err) {
      console.error(err)
      setError("An error occurred during ingestion.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>CityWok Episode Identifier</h1>
        <p>Find which South Park episode your clip is from.</p>
      </header>

      <div className="controls">
        <button 
          onClick={() => setMode('identify')} 
          className={mode === 'identify' ? 'active' : ''}
        >
          Identify Clip
        </button>
        <button 
          onClick={() => setMode('ingest')} 
          className={mode === 'ingest' ? 'active' : ''}
        >
          Admin: Ingest Episode
        </button>
      </div>

      <main className="card">
        <div className="upload-section">
          <input 
            type="file" 
            accept="video/*" 
            onChange={handleFileChange} 
          />
          
          {mode === 'ingest' && (
            <input 
              type="text" 
              placeholder="Episode ID (e.g., S01E01)" 
              value={episodeId}
              onChange={(e) => setEpisodeId(e.target.value)}
              style={{marginTop: '10px', display: 'block'}}
            />
          )}

          <button 
            onClick={mode === 'identify' ? handleIdentify : handleIngest} 
            disabled={!file || loading}
            className="action-btn"
          >
            {loading ? 'Processing...' : (mode === 'identify' ? 'Identify Episode' : 'Ingest Episode')}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            {mode === 'identify' ? (
              <>
                {result.match_found ? (
                  <div className="success-match">
                    <h3>Match Found!</h3>
                    <p className="episode-title">Episode: {result.episode}</p>
                    <p className="timestamp">Time: {result.timestamp}</p>
                    <div className="details">
                      <small>Confidence: {result.details.confidence}</small>
                      <br/>
                      <small>Method: {result.details.method}</small>
                    </div>
                  </div>
                ) : (
                  <div className="no-match">
                    <h3>No Match Found</h3>
                    <p>We couldn't identify this clip with high confidence.</p>
                  </div>
                )}
              </>
            ) : (
              <div className="success-match">
                <p>{result.message}</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
