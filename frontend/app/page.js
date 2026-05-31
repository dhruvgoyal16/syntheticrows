"use client"
import { useState, useRef } from "react"

// ─── Constants ────────────────────────────────────────────────────────────────

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

const ISSUE_EXPLANATIONS = {
  fill_missing: (col) => ({
    what: `${col} has missing values that could confuse the generation model.`,
    impact: "Missing values cause the model to learn incomplete patterns, producing synthetic data with gaps or unrealistic distributions.",
    action: "We'll fill these with the column median — a safe, statistically sound replacement that preserves the data's natural center."
  }),
  fix_zeros: (col) => ({
    what: `${col} has an unusually high number of zeros that likely represent missing data, not actual zero values.`,
    impact: "If left unfixed, the model will learn that zeros are common and generate them frequently — creating biologically or logically impossible values in your synthetic data.",
    action: "We'll replace these zeros with the column median calculated from real non-zero values, preserving the true distribution."
  }),
  cap_outliers: (col) => ({
    what: `${col} has extreme values that sit far outside the normal range of the data.`,
    impact: "Outliers distort what the model learns as 'normal', causing it to occasionally generate unrealistic extreme values in synthetic rows.",
    action: "We'll cap values at ±3 standard deviations — keeping the natural spread while removing values that would mislead the model."
  }),
  drop_column: (col) => ({
    what: `${col} appears to be an ID, constant, or high-cardinality column with too many unique values to synthesize meaningfully.`,
    impact: "Synthesizing ID or constant columns adds noise without value — the model wastes capacity learning patterns that don't exist.",
    action: "We'll remove this column before generation. It will not appear in your synthetic dataset."
  })
}

const SCORE_INTERPRETATION = (score, grade, stats) => {
  const { distinguishability_score, statistical_score, coverage_score } = stats

  let overall = ""
  if (score >= 80) {
    overall = "Your synthetic data is excellent. A model trained on this data should perform comparably to one trained on your real data. You can confidently use this for ML training, testing, and augmentation."
  } else if (score >= 60) {
    overall = "Your synthetic data is good and usable for most ML tasks. Some statistical patterns may differ slightly from your real data, but the overall distributions are well-preserved."
  } else {
    overall = "Your synthetic data is fair. Consider approving more data quality fixes before regenerating, or try with a cleaner dataset. The data can still be useful for prototyping."
  }

  let weakest = ""
  if (distinguishability_score < 60) {
    weakest = "The distinguishability score is low — a classifier can tell real from synthetic data. This usually means inter-column correlations aren't fully preserved. Try approving more fixes before regenerating."
  } else if (statistical_score < 60) {
    weakest = "The statistical similarity score is low — some column distributions differ significantly from your real data. Check the column report below to see which columns need attention."
  } else if (coverage_score < 60) {
    weakest = "The coverage score is low — synthetic data doesn't cover the full range of your real data. Try generating more rows to improve coverage."
  }

  return { overall, weakest }
}

const GENERATION_STAGES = [
  { id: 1, label: "Profiling dataset", desc: "Analysing column types and relationships" },
  { id: 2, label: "Applying fixes", desc: "Cleaning data based on your selections" },
  { id: 3, label: "Training model", desc: "Learning patterns from your data" },
  { id: 4, label: "Generating rows", desc: "Creating realistic synthetic samples" },
  { id: 5, label: "Scoring quality", desc: "Evaluating realism across three metrics" },
]

// ─── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState(null)
  const [summary, setSummary] = useState(null)
  const [issues, setIssues] = useState([])
  const [fixes, setFixes] = useState([])
  const [step, setStep] = useState("upload")
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [generationStage, setGenerationStage] = useState(0)
  const [error, setError] = useState(null)
  const [numRows, setNumRows] = useState(300)
  const [result, setResult] = useState(null)
  const [expandedIssue, setExpandedIssue] = useState(null)
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
      setError("Could not connect to backend. Make sure the backend server is running.")
    } finally {
      setLoading(false)
    }
  }

  const toggleFix = (index) => {
    setFixes((prev) =>
      prev.map((f, i) => i === index ? { ...f, approved: !f.approved } : f)
    )
  }

  const simulateStages = () => {
    const timings = [800, 1500, 4000, 3000, 2000]
    let cumulative = 0
    timings.forEach((delay, i) => {
      cumulative += delay
      setTimeout(() => setGenerationStage(i + 1), cumulative)
    })
  }

  const handleGenerate = async () => {
    if (!file) return
    setGenerating(true)
    setError(null)
    setGenerationStage(0)
    simulateStages()

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
      setError(
        err.message?.includes("500")
          ? "Generation failed. Your dataset may have unsupported column types. Try approving more fixes and regenerating."
          : err.message || "Something went wrong. Please try again."
      )
    } finally {
      setGenerating(false)
      setGenerationStage(0)
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

  const previewRows = result?.csv_data
    ? (() => {
        const lines = result.csv_data.trim().split("\n")
        const headers = lines[0].split(",")
        const rows = lines.slice(1, 6).map(line => line.split(","))
        return { headers, rows }
      })()
    : null

  const interpretation = result
    ? SCORE_INTERPRETATION(result.realism_score, result.grade, result)
    : null

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

        {/* Upload Box */}
        <div
          onClick={() => !generating && inputRef.current.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            handleFile(e.dataTransfer.files[0])
          }}
          className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all
            ${generating ? "border-gray-700 cursor-not-allowed opacity-50" : "border-violet-500 cursor-pointer hover:bg-violet-500/5"}`}
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
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

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

        {/* Dataset Info Banner */}
        {summary && step !== "upload" && (
          <div className="bg-gray-900 rounded-xl p-4 flex items-center justify-between">
            <div>
              <p className="text-white text-sm font-semibold">{summary.filename}</p>
              <p className="text-gray-500 text-xs mt-0.5">
                {summary.rows} rows · {summary.columns} columns · {summary.size_category}
                {summary.is_imbalanced && <span className="text-yellow-400 ml-2">⚠ Imbalanced</span>}
                {summary.has_datetime && <span className="text-blue-400 ml-2">📅 Time series</span>}
              </p>
            </div>
            <span className="text-violet-400 text-xs bg-violet-500/10 px-3 py-1 rounded-full">
              {summary.dataset_type}
            </span>
          </div>
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
              We found {issues.length} potential issues in your dataset. Review each one and toggle to approve or skip the fix.
            </p>

            <div className="space-y-3 mb-6">
              {issues.map((issue, i) => {
                const explanation = ISSUE_EXPLANATIONS[issue.fix_type]?.(issue.column)
                const isExpanded = expandedIssue === i

                return (
                  <div
                    key={i}
                    className={`border rounded-xl p-4 transition-all ${fixes[i]?.approved
                      ? severityColor[issue.severity]
                      : "border-gray-700 bg-gray-800/50 opacity-60"
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

                        {/* Explanation */}
                        {explanation && (
                          <div className="mt-2">
                            <button
                              onClick={() => setExpandedIssue(isExpanded ? null : i)}
                              className="text-violet-400 text-xs hover:text-violet-300 transition-colors"
                            >
                              {isExpanded ? "▼ Hide explanation" : "▶ Why does this matter?"}
                            </button>

                            {isExpanded && (
                              <div className="mt-2 space-y-2">
                                <div className="bg-gray-900/50 rounded-lg p-3">
                                  <p className="text-gray-300 text-xs font-semibold mb-1">🔍 What we found</p>
                                  <p className="text-gray-400 text-xs">{explanation.what}</p>
                                </div>
                                <div className="bg-red-500/5 rounded-lg p-3">
                                  <p className="text-red-400 text-xs font-semibold mb-1">⚠ Impact if ignored</p>
                                  <p className="text-gray-400 text-xs">{explanation.impact}</p>
                                </div>
                                <div className="bg-green-500/5 rounded-lg p-3">
                                  <p className="text-green-400 text-xs font-semibold mb-1">✓ What we'll do</p>
                                  <p className="text-gray-400 text-xs">{explanation.action}</p>
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        <p className="text-gray-600 text-xs mt-2">
                          Fix: {issue.recommendation}
                        </p>
                      </div>

                      {/* Toggle */}
                      <button
                        onClick={() => toggleFix(i)}
                        className={`shrink-0 w-10 h-6 rounded-full transition-all ${fixes[i]?.approved ? "bg-violet-600" : "bg-gray-700"}`}
                      >
                        <div className={`w-4 h-4 bg-white rounded-full mx-auto transition-all ${fixes[i]?.approved ? "translate-x-2" : "-translate-x-2"}`} />
                      </button>
                    </div>
                  </div>
                )
              })}
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

            {summary?.is_imbalanced && (
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-3 mb-4">
                <p className="text-yellow-300 text-sm font-semibold">⚠ Imbalanced dataset detected</p>
                <p className="text-yellow-400/70 text-xs mt-1">
                  We detected a class imbalance ({Math.round(summary.imbalance_ratio * 100)}% / {Math.round((1 - summary.imbalance_ratio) * 100)}%). SynthIQ will automatically preserve this ratio in your synthetic data using conditional generation — so your model trains on correctly distributed data.
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
                    ⚠ Your dataset has {summary.rows} rows. Generating more than {summary.rows * 2} rows may reduce quality. We'll automatically cap at {summary.rows * 2} for best results.
                  </p>
                </div>
              )}
            </div>

            {/* Progress Indicator */}
            {generating && (
              <div className="bg-gray-800 rounded-xl p-4 mb-4">
                <p className="text-violet-400 text-sm font-semibold mb-3">Generating your dataset...</p>
                <div className="space-y-2">
                  {GENERATION_STAGES.map((stage) => {
                    const isDone = generationStage > stage.id
                    const isActive = generationStage === stage.id
                    return (
                      <div key={stage.id} className="flex items-center gap-3">
                        <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs shrink-0
                          ${isDone ? "bg-green-500" : isActive ? "bg-violet-500 animate-pulse" : "bg-gray-700"}`}>
                          {isDone ? "✓" : isActive ? "●" : "○"}
                        </div>
                        <div>
                          <p className={`text-xs font-semibold ${isDone ? "text-green-400" : isActive ? "text-white" : "text-gray-600"}`}>
                            {stage.label}
                          </p>
                          {isActive && (
                            <p className="text-gray-500 text-xs">{stage.desc}</p>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            <button
              onClick={handleGenerate}
              disabled={generating}
              className="w-full bg-violet-600 hover:bg-violet-700 text-white font-semibold px-8 py-3 rounded-xl transition-all disabled:opacity-50"
            >
              {generating ? "Generating..." : "⚡ Generate Synthetic Dataset"}
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
              {interpretation && (
                <p className="text-gray-400 text-xs mt-3 text-left leading-relaxed">
                  {interpretation.overall}
                </p>
              )}
              {interpretation?.weakest && (
                <p className="text-yellow-400 text-xs mt-2 text-left leading-relaxed">
                  💡 {interpretation.weakest}
                </p>
              )}
            </div>

            {/* Three Sub-scores */}
            <div className="bg-gray-800 rounded-xl p-4 space-y-3">
              <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide">Score Breakdown</p>
              <div className="space-y-3">
                {[
                  {
                    label: "Statistical Similarity",
                    value: result.statistical_score,
                    desc: "Do column distributions match your real data?",
                    weight: "50%",
                    interpretation: result.statistical_score >= 80
                      ? "Column distributions closely match your real data."
                      : result.statistical_score >= 60
                      ? "Most distributions match but some columns differ."
                      : "Several column distributions differ significantly."
                  },
                  {
                    label: "Coverage",
                    value: result.coverage_score,
                    desc: "Does synthetic data cover the full value range?",
                    weight: "30%",
                    interpretation: result.coverage_score >= 80
                      ? "Synthetic data covers the full range of real values."
                      : "Some value ranges from real data aren't fully covered."
                  },
                  {
                    label: "Distinguishability",
                    value: result.distinguishability_score,
                    desc: "Can a classifier tell real from synthetic?",
                    weight: "20%",
                    interpretation: result.distinguishability_score >= 80
                      ? "A classifier cannot reliably distinguish synthetic from real."
                      : result.distinguishability_score >= 60
                      ? "Some patterns differ but data is still usable."
                      : "Inter-column correlations could be stronger."
                  }
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
                        className={`h-1.5 rounded-full ${
                          metric.value >= 80 ? "bg-green-400" :
                          metric.value >= 60 ? "bg-yellow-400" : "bg-red-400"
                        }`}
                        style={{ width: `${metric.value}%` }}
                      />
                    </div>
                    <p className="text-gray-600 text-xs mt-1">{metric.interpretation}</p>
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
                        }`}>
                          {col.grade === "Excellent" ? "✓" : col.grade === "Good" ? "~" : "!"}
                        </span>
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

            {/* Dataset Preview */}
            {previewRows && (
              <div className="bg-gray-800 rounded-xl p-4">
                <p className="text-gray-400 text-xs font-semibold uppercase tracking-wide mb-3">
                  Synthetic Data Preview (first 5 rows)
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr>
                        {previewRows.headers.map((h) => (
                          <th key={h} className="text-gray-500 font-semibold pb-2 pr-3 text-left whitespace-nowrap">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewRows.rows.map((row, i) => (
                        <tr key={i} className="border-t border-gray-700">
                          {row.map((cell, j) => (
                            <td key={j} className="text-gray-300 py-1.5 pr-3 whitespace-nowrap">
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
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

            {result.capped && (
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-3">
                <p className="text-yellow-400 text-xs">
                  ⚠ Row count was capped at {result.max_recommended} (2× your real data) to maintain quality.
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