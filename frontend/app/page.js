"use client"
import { useState, useRef } from "react"

const severityColor = {
  high: "border-red-500/50 bg-red-500/5",
  medium: "border-yellow-500/50 bg-yellow-500/5",
}

const severityBadge = {
  high: "bg-red-500/20 text-red-400",
  medium: "bg-yellow-500/20 text-yellow-400",
}

const scoreColor = {
  green: "text-green-400",
  yellow: "text-yellow-400",
  red: "text-red-400"
}

export default function Home() {
  const [file, setFile] = useState(null)
  const [summary, setSummary] = useState(null)
  const [issues, setIssues] = useState([])
  const [fixes, setFixes] = useState([])
  const [step, setStep] = useState("upload") // upload → analyse → quality → generate → result
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState(null)
  const [numRows, setNumRows] = useState(300)
  const [result, setResult] = useState(null)
  const inputRef = useRef(null)

  const handleFile = (selectedFile) => {
    if (!selectedFile.name.endsWith(".csv")) {
      setError("Only CSV files are allowed")
      return
    }
    setFile(selectedFile)
    setError(null)
    setSummary(null)
    setIssues([])
    setFixes([])
    setResult(null)
    setStep("upload")
  }

  const handleAnalyse = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch("http://localhost:8000/analyse", {
        method: "POST",
        body: formData,
      })
      const data = await res.json()
      setSummary(data)

      // Default all fixes to approved
      const defaultFixes = data.issues.map((issue) => ({
        column: issue.column,
        issue: issue.issue,
        fix_type: issue.fix_type,
        approved: true
      }))
      setIssues(data.issues)
      setFixes(defaultFixes)
      setStep(data.issues.length > 0 ? "quality" : "generate")
    } catch (err) {
      setError("Could not connect to backend. Is it running?")
    } finally {
      setLoading(false)
    }
  }

  const toggleFix = (index) => {
    setFixes((prev) =>
      prev.map((f, i) => i === index ? { ...f, approved: !f.approved } : f)
    )
  }

  const handleGenerate = async () => {
    if (!file) return
    setGenerating(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)
    formData.append("fixes", JSON.stringify(fixes))

    try {
      const res = await fetch(`http://localhost:8000/generate-with-score?num_rows=${numRows}`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail)
      }

      const data = await res.json()
      setResult(data)
      setStep("result")

    } catch (err) {
      setError(err.message || "Generation failed. Please try again.")
    } finally {
      setGenerating(false)
    }
  }

  const handleDownload = () => {
    if (!result?.csv_data) return
    const blob = new Blob([result.csv_data], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `synthetic_${file.name}`
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const approvedCount = fixes.filter(f => f.approved).length

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

      {/* Step Indicator */}
      <div className="flex items-center gap-2 mb-8 text-sm">
        {["Upload", "Quality Check", "Generate", "Result"].map((s, i) => {
          const stepKeys = ["upload", "quality", "generate", "result"]
          const current = stepKeys.indexOf(step)
          const isActive = i === current
          const isDone = i < current
          return (
            <div key={s} className="flex items-center gap-2">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                ${isDone ? "bg-violet-600 text-white" : isActive ? "bg-violet-500 text-white" : "bg-gray-800 text-gray-500"}`}>
                {isDone ? "✓" : i + 1}
              </div>
              <span className={isActive ? "text-white" : isDone ? "text-violet-400" : "text-gray-600"}>
                {s}
              </span>
              {i < 3 && <span className="text-gray-700">→</span>}
            </div>
          )
        })}
      </div>

      <div className="w-full max-w-lg space-y-4">

        {/* Upload Box — always visible */}
        <div
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            handleFile(e.dataTransfer.files[0])
          }}
          className="border-2 border-dashed border-violet-500 rounded-2xl p-10 text-center cursor-pointer hover:bg-violet-500/5 transition-all"
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
        {error && <p className="text-red-400 text-sm text-center">{error}</p>}

        {/* Analyse Button */}
        {file && step === "upload" && (
          <button
            onClick={handleAnalyse}
            disabled={loading}
            className="w-full bg-violet-600 hover:bg-violet-700 text-white font-semibold px-8 py-3 rounded-xl transition-all disabled:opacity-50"
          >
            {loading ? "Analysing dataset..." : "Analyse Dataset"}
          </button>
        )}

        {/* Data Quality Report */}
        {step === "quality" && issues.length > 0 && (
          <div className="bg-gray-900 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-violet-400 font-semibold text-lg">Data Quality Report</h2>
              <span className="text-xs bg-gray-800 text-gray-400 px-3 py-1 rounded-full">
                {approvedCount} fix{approvedCount !== 1 ? "es" : ""} selected
              </span>
            </div>
            <p className="text-gray-500 text-sm mb-4">
              We found {issues.length} potential issues. Toggle to approve or skip each fix.
            </p>

            <div className="space-y-3 mb-6">
              {issues.map((issue, i) => (
                <div
                  key={i}
                  className={`border rounded-xl p-4 transition-all ${fixes[i]?.approved
                    ? severityColor[issue.severity]
                    : "border-gray-700 bg-gray-800/50 opacity-50"
                    }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-white font-semibold text-sm">{issue.column}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${severityBadge[issue.severity]}`}>
                          {issue.severity}
                        </span>
                      </div>
                      <p className="text-gray-400 text-xs">{issue.issue}</p>
                      <p className="text-gray-500 text-xs mt-1">
                        Fix: {issue.recommendation}
                      </p>
                    </div>
                    <button
                      onClick={() => toggleFix(i)}
                      className={`shrink-0 w-10 h-6 rounded-full transition-all ${fixes[i]?.approved ? "bg-violet-600" : "bg-gray-700"}`}
                    >
                      <div className={`w-4 h-4 bg-white rounded-full mx-auto transition-all ${fixes[i]?.approved ? "translate-x-2" : "-translate-x-2"}`} />
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={() => setStep("generate")}
              className="w-full bg-violet-600 hover:bg-violet-700 text-white font-semibold px-8 py-3 rounded-xl transition-all"
            >
              Continue with {approvedCount} fix{approvedCount !== 1 ? "es" : ""} →
            </button>
          </div>
        )}

        {/* Generate Controls */}
        {step === "generate" && (
          <div className="bg-gray-900 rounded-2xl p-6">
            <h2 className="text-violet-400 font-semibold text-lg mb-4">Generate Synthetic Data</h2>

            {approvedCount > 0 && (
              <div className="bg-violet-500/10 border border-violet-500/30 rounded-xl p-3 mb-4">
                <p className="text-violet-300 text-sm">
                  ✓ {approvedCount} data quality fix{approvedCount !== 1 ? "es" : ""} will be applied before generation
                </p>
              </div>
            )}

            <div className="mb-6">
              <label className="text-gray-400 text-sm block mb-2">
                Rows to generate: <span className="text-white font-bold">{numRows}</span>
              </label>
              <input
                type="range"
                min="100"
                max="1000"
                step="50"
                value={numRows}
                onChange={(e) => setNumRows(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
              <div className="flex justify-between text-gray-600 text-xs mt-1">
                <span>100</span>
                <span>1000 (free limit)</span>
              </div>
              {summary && numRows > summary.rows * 2 && (
                <div className="mt-3 bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-3">
                  <p className="text-yellow-400 text-xs">
                    ⚠️ Your dataset has {summary.rows} rows. Generating more than {summary.rows * 2} rows may reduce quality. We'll automatically cap at {summary.rows * 2} for best results.
                  </p>
                </div>
              )}
            </div>

            <button
              onClick={handleGenerate}
              disabled={generating}
              className="w-full bg-violet-600 hover:bg-violet-700 text-white font-semibold px-8 py-3 rounded-xl transition-all disabled:opacity-50"
            >
              {generating ? "Generating... (this takes ~2 mins)" : "⚡ Generate Synthetic Dataset"}
            </button>
          </div>
        )}

        {/* Result */}
        {step === "result" && result && (
          <div className="bg-gray-900 rounded-2xl p-6 space-y-4">
            <h2 className="text-violet-400 font-semibold text-lg">Generation Complete</h2>

            {/* Main Score */}
            <div className="bg-gray-800 rounded-xl p-6 text-center">
              <p className="text-gray-400 text-sm mb-1">Realism Score</p>
              <p className={`text-6xl font-bold ${scoreColor[result.color]}`}>
                {result.realism_score}
              </p>
              <p className={`text-sm font-semibold mt-1 ${scoreColor[result.color]}`}>
                {result.grade}
              </p>
              <p className="text-gray-600 text-xs mt-3">
                Weighted average of three quality metrics below
              </p>
            </div>

            {/* Three Sub-scores */}
            <div className="bg-gray-800 rounded-xl p-4 space-y-3">
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide">Score Breakdown</p>

              <div className="space-y-2">
                {[
                  { label: "Distinguishability", value: result.distinguishability_score, desc: "Can a classifier tell real from synthetic?", weight: "20%" },
                  { label: "Statistical Similarity", value: result.statistical_score, desc: "Do distributions match column by column?", weight: "50%" },
                  { label: "Coverage", value: result.coverage_score, desc: "Does synthetic data cover the full data range?", weight: "30%" }
                ].map((metric) => (
                  <div key={metric.label}>
                    <div className="flex justify-between items-center mb-1">
                      <div>
                        <span className="text-white text-xs font-semibold">{metric.label}</span>
                        <span className="text-gray-600 text-xs ml-2">({metric.weight})</span>
                      </div>
                      <span className={`text-xs font-bold ${
                        metric.value >= 80 ? "text-green-400" :
                        metric.value >= 60 ? "text-yellow-400" : "text-red-400"
                      }`}>{metric.value}</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full transition-all ${
                          metric.value >= 80 ? "bg-green-400" :
                          metric.value >= 60 ? "bg-yellow-400" : "bg-red-400"
                        }`}
                        style={{ width: `${metric.value}%` }}
                      />
                    </div>
                    <p className="text-gray-600 text-xs mt-0.5">{metric.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Column Quality Report */}
            {result.column_quality && result.column_quality.length > 0 && (
              <div className="bg-gray-800 rounded-xl p-4">
                <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide mb-3">
                  Column Quality Report
                </p>
                <div className="space-y-2">
                  {result.column_quality.map((col) => (
                    <div key={col.column} className="flex items-center justify-between">
                      <span className="text-gray-300 text-xs truncate flex-1">{col.column}</span>
                      <div className="flex items-center gap-2 ml-2">
                        <div className="w-16 bg-gray-700 rounded-full h-1">
                          <div
                            className={`h-1 rounded-full ${
                              col.score >= 80 ? "bg-green-400" :
                              col.score >= 60 ? "bg-yellow-400" : "bg-red-400"
                            }`}
                            style={{ width: `${col.score}%` }}
                          />
                        </div>
                        <span className={`text-xs font-semibold w-8 text-right ${
                          col.grade === "Excellent" ? "text-green-400" :
                          col.grade === "Good" ? "text-yellow-400" :
                          col.grade === "Fair" ? "text-orange-400" : "text-red-400"
                        }`}>{col.grade === "Excellent" ? "✓" : col.grade === "Good" ? "~" : "!"}</span>
                        <span className="text-gray-500 text-xs w-8 text-right">{col.score}</span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex gap-4 mt-3 pt-3 border-t border-gray-700">
                  <span className="text-green-400 text-xs">✓ Excellent (80+)</span>
                  <span className="text-yellow-400 text-xs">~ Good (60+)</span>
                  <span className="text-orange-400 text-xs">! Fair/Poor</span>
                </div>
              </div>
            )}

            {/* Stats Row */}
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gray-800 rounded-xl p-4 text-center">
                <p className="text-gray-400 text-xs">Rows Generated</p>
                <p className="text-white text-xl font-bold">{result.rows_generated}</p>
              </div>
              <div className="bg-gray-800 rounded-xl p-4 text-center">
                <p className="text-gray-400 text-xs">Fixes Applied</p>
                <p className="text-white text-xl font-bold">{approvedCount}</p>
              </div>
              <div className="bg-gray-800 rounded-xl p-4 text-center">
                <p className="text-gray-400 text-xs">Model Used</p>
                <p className="text-violet-400 text-xs font-bold mt-1">{result.model_used}</p>
              </div>
            </div>

            {/* Capped warning */}
            {result.capped && (
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-3">
                <p className="text-yellow-400 text-xs">
                  ⚠️ Row count was capped at {result.max_recommended} (2× your real data) to maintain quality.
                </p>
              </div>
            )}

            {/* Download */}
            <button
              onClick={handleDownload}
              className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold px-8 py-3 rounded-xl transition-all"
            >
              ⬇️ Download Synthetic Dataset
            </button>

            <button
              onClick={() => {
                setFile(null)
                setSummary(null)
                setIssues([])
                setFixes([])
                setResult(null)
                setStep("upload")
              }}
              className="w-full bg-gray-800 hover:bg-gray-700 text-gray-400 font-semibold px-8 py-3 rounded-xl transition-all"
            >
              Start Over
            </button>
          </div>
        )}

      </div>

      <p className="text-gray-600 text-sm mt-8">
        Supports CSV files · Free up to 1000 rows
      </p>

    </main>
  )
}