"use client"
import { useState, useRef } from "react"

export default function Home() {
  const [file, setFile] = useState(null)
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState(null)
  const [numRows, setNumRows] = useState(300)
  const inputRef = useRef(null)

  const handleFile = (selectedFile) => {
    if (!selectedFile.name.endsWith(".csv")) {
      setError("Only CSV files are allowed")
      return
    }
    setFile(selectedFile)
    setError(null)
    setSummary(null)
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      })
      const data = await res.json()
      setSummary(data)
    } catch (err) {
      setError("Could not connect to backend. Is it running?")
    } finally {
      setLoading(false)
    }
  }

  const handleGenerate = async () => {
    if (!file) return
    setGenerating(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch(`http://localhost:8000/generate?num_rows=${numRows}`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail)
      }

      // Trigger file download
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `synthetic_${file.name}`
      a.click()
      window.URL.revokeObjectURL(url)

    } catch (err) {
      setError(err.message || "Generation failed. Please try again.")
    } finally {
      setGenerating(false)
    }
  }

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center px-4 py-16">

      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-white mb-4">
          Synth<span className="text-violet-500">IQ</span>
        </h1>
        <p className="text-gray-400 text-xl max-w-xl">
          Upload your small dataset. Get back a larger, realistic version that trains better models.
        </p>
      </div>

      {/* Upload Box */}
      <div
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          handleFile(e.dataTransfer.files[0])
        }}
        className="border-2 border-dashed border-violet-500 rounded-2xl p-12 text-center w-full max-w-lg cursor-pointer hover:bg-violet-500/5 transition-all"
      >
        <p className="text-4xl mb-4">📂</p>
        {file ? (
          <p className="text-violet-400 font-semibold">{file.name}</p>
        ) : (
          <>
            <p className="text-white font-semibold text-lg">Drop your CSV here</p>
            <p className="text-gray-500 text-sm mt-2">or click to browse</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {/* Error */}
      {error && (
        <p className="text-red-400 text-sm mt-4">{error}</p>
      )}

      {/* Analyse Button */}
      {file && !summary && (
        <button
          onClick={handleUpload}
          disabled={loading}
          className="mt-6 bg-violet-600 hover:bg-violet-700 text-white font-semibold px-8 py-3 rounded-xl transition-all disabled:opacity-50"
        >
          {loading ? "Analysing..." : "Analyse Dataset"}
        </button>
      )}

      {/* Summary */}
      {summary && (
        <div className="mt-8 w-full max-w-lg space-y-4">

          {/* Stats */}
          <div className="bg-gray-900 rounded-2xl p-6">
            <h2 className="text-violet-400 font-semibold text-lg mb-4">Dataset Summary</h2>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-gray-800 rounded-xl p-4">
                <p className="text-gray-400 text-sm">Rows</p>
                <p className="text-white text-2xl font-bold">{summary.rows}</p>
              </div>
              <div className="bg-gray-800 rounded-xl p-4">
                <p className="text-gray-400 text-sm">Columns</p>
                <p className="text-white text-2xl font-bold">{summary.columns}</p>
              </div>
            </div>
            <div className="bg-gray-800 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-2">Columns Detected</p>
              <div className="flex flex-wrap gap-2">
                {summary.column_names.map((col) => (
                  <span key={col} className="bg-violet-500/20 text-violet-300 text-xs px-3 py-1 rounded-full">
                    {col}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Generate Controls */}
          <div className="bg-gray-900 rounded-2xl p-6">
            <h2 className="text-violet-400 font-semibold text-lg mb-4">Generate Synthetic Data</h2>
            <div className="mb-4">
              <label className="text-gray-400 text-sm block mb-2">
                Number of rows to generate: <span className="text-white font-bold">{numRows}</span>
              </label>
              <input
                type="range"
                min="100"
                max="500"
                step="50"
                value={numRows}
                onChange={(e) => setNumRows(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
              <div className="flex justify-between text-gray-600 text-xs mt-1">
                <span>100</span>
                <span>500 (free limit)</span>
              </div>
            </div>
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="w-full bg-violet-600 hover:bg-violet-700 text-white font-semibold px-8 py-3 rounded-xl transition-all disabled:opacity-50"
            >
              {generating ? "Generating... (this takes ~1 min)" : "⚡ Generate Synthetic Dataset"}
            </button>
          </div>

        </div>
      )}

      <p className="text-gray-600 text-sm mt-8">
        Supports CSV files · Free up to 500 rows
      </p>

    </main>
  )
}