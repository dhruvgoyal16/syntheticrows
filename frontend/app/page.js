"use client"
import { useState, useRef, useEffect } from "react"

// ─── Constants ────────────────────────────────────────────────────────────────

const ISSUE_EXPLANATIONS = {
  fill_missing: (col) => ({
    what: `${col} has missing values that could confuse the generation model.`,
    impact: "Missing values cause the model to learn incomplete patterns, producing synthetic data with gaps or unrealistic distributions.",
    action: "We'll fill these with the column median — a safe, statistically sound replacement that preserves the data's natural center."
  }),
  fix_zeros: (col) => ({
    what: `${col} has an unusually high number of zeros that likely represent missing data, not actual zero values.`,
    impact: "If left unfixed, the model learns that zeros are common and generates them often — creating impossible values in your synthetic data.",
    action: "We'll replace these zeros with the column median from real non-zero values, preserving the true distribution."
  }),
  cap_outliers: (col) => ({
    what: `${col} has extreme values far outside the normal range of the data.`,
    impact: "Outliers distort what the model learns as 'normal', causing occasional unrealistic extreme values in synthetic rows.",
    action: "We'll cap values at ±3 standard deviations — keeping the natural spread while removing misleading extremes."
  }),
  drop_column: (col) => ({
    what: `${col} looks like an ID, constant, or high-cardinality column with too many unique values to synthesize meaningfully.`,
    impact: "Synthesizing such columns adds noise without value — the model wastes capacity on patterns that don't exist.",
    action: "We'll remove this column before generation. It won't appear in your synthetic dataset."
  })
}

const SCORE_INTERPRETATION = (score, stats, summary) => {
  const { distinguishability_score, statistical_score, coverage_score } = stats
  let overall = ""
  if (score >= 80) overall = "Your synthetic data is excellent. A model trained on it should perform comparably to one trained on your real data — ready for ML training, testing, and augmentation."
  else if (score >= 60) overall = "Your synthetic data is good and usable for most ML tasks. Overall distributions are well-preserved, though a few statistical patterns differ slightly from your real data."
  else {
    // Low score — try to give a SPECIFIC, honest diagnosis instead of a generic line.
    const ratio = summary?.imbalance_ratio
    const extremeImbalance = summary?.is_imbalanced && typeof ratio === "number" && ratio >= 0.9
    if (extremeImbalance) {
      const majority = Math.round(ratio * 100)
      overall = `Your synthetic data scored low, and the main reason is extreme class imbalance — one class makes up about ${majority}% of your data. With so few examples of the minority class, the model can't learn its patterns well, so the synthetic data struggles to reproduce them. The most effective fix here isn't a setting — it's collecting more real examples of the rare class. You can still use this data for prototyping, but treat the minority class with caution.`
    } else if (coverage_score < 30) {
      overall = "Your synthetic data scored low mainly because of poor coverage — the synthetic rows don't span the full range of your real values. This usually means your data has extreme values or a heavy-tailed distribution that's hard to reproduce. Try approving the outlier-capping fixes, or check whether a few extreme rows are skewing the column. It can still be useful for prototyping."
    } else if (statistical_score < 40) {
      overall = "Your synthetic data scored low because the column distributions differ noticeably from your real data — check the column report below to see which columns drifted most. Approving more data-quality fixes and regenerating often helps. It can still be useful for prototyping."
    } else {
      overall = "Your synthetic data is fair. Consider approving more data quality fixes before regenerating, or try a cleaner dataset. It can still be useful for prototyping."
    }
  }
  let weakest = ""
  if (distinguishability_score < 60) weakest = "Distinguishability is low — a classifier can tell real from synthetic. Inter-column correlations aren't fully preserved; try approving more fixes."
  else if (statistical_score < 60) weakest = "Statistical similarity is low — some column distributions differ. Check the column report below."
  else if (coverage_score < 60) weakest = "Coverage is low — synthetic data doesn't span the full range of real values. Try generating more rows."
  return { overall, weakest }
}

const GENERATION_STAGES = [
  { id: 1, label: "Profiling dataset" },
  { id: 2, label: "Applying fixes" },
  { id: 3, label: "Training model" },
  { id: 4, label: "Generating rows" },
  { id: 5, label: "Scoring quality" },
]

// ─── SWAP THIS with your real donation link (Razorpay page / UPI link / etc.) ───
const DONATE_URL = "https://razorpay.me/@dhruvgoyal"

const FOOTER_QUIPS = [
  "Real people's data leaked: 0. That's kind of the whole point.",
  "Marketing budget: zero rupees. You found us anyway. Witchcraft.",
  "We built an entire honesty metric just so we couldn't lie to you about quality.",
  "No rows were harmed in the making of this dataset.",
  "Your tiny dataset is safe with us. We don't judge. (The model might.)",
  "Powered by maths, caffeine, and one very tired founder.",
  "Synthetic data: all the insights, none of the privacy lawsuits.",
  "We read the SDV docs so you don't have to. You're welcome.",
  "Trained on your 50 rows. Complained zero times.",
  "100% synthetic. 0% \"we'll totally delete your data later, trust us.\"",
]

const sColor = (v) => v >= 80 ? "var(--good)" : v >= 60 ? "var(--warn)" : "var(--bad)"
const sWash = (v) => v >= 80 ? "var(--good-wash)" : v >= 60 ? "var(--warn-wash)" : "var(--bad-wash)"
const tstrScore = (c) => c === "green" ? 90 : c === "yellow" ? 70 : 40
// fills a range input yellow from the left up to the current value
const sliderFill = (val, min = 100, max = 1000) => {
  const pct = ((val - min) / (max - min)) * 100
  return { background: `linear-gradient(to right, var(--yellow) 0%, var(--yellow) ${pct}%, var(--line-2) ${pct}%, var(--line-2) 100%)` }
}

// ─── Heatmap ─────────────────────────────────────────────────────────────────

function Heatmap({ title, columns, matrix }) {
  const n = matrix.length, cs = 46, pad = 30, size = n * cs + pad
  const labels = columns.map(c => c.length > 5 ? c.slice(0, 5) : c)
  const cell = (v) => {
    if (v == null || isNaN(v)) return "#e8e8e4"
    const vv = Math.max(-1, Math.min(1, v))
    if (vv > 0) { const k = Math.round(vv * 200); return `rgb(${255 - k},${255 - Math.round(k * 0.25)},${255 - k})` }
    const k = Math.round(-vv * 200); return `rgb(255,${255 - k},${255 - k})`
  }
  return (
    <div className="sq-heat">
      <h4>{title}</h4>
      <svg viewBox={`0 0 ${size} ${size}`}>
        {labels.map((l, i) => <text key={`r${i}`} x={pad - 4} y={pad + i * cs + cs / 2 + 3} textAnchor="end" fontSize="10" fill="#777">{l}</text>)}
        {labels.map((l, i) => <text key={`c${i}`} x={pad + i * cs + cs / 2} y={pad - 5} textAnchor="middle" fontSize="10" fill="#777">{l}</text>)}
        {matrix.map((row, i) => row.map((v, j) => (
          <g key={`${i}-${j}`}>
            <rect x={pad + j * cs} y={pad + i * cs} width={cs - 2} height={cs - 2} rx="3" fill={cell(v)} />
            <text x={pad + j * cs + cs / 2 - 1} y={pad + i * cs + cs / 2 + 3} textAnchor="middle" fontSize="9.5" fill={Math.abs(v) > 0.6 ? "#0f0f0f" : "#777"} fontWeight={Math.abs(v) > 0.6 ? 700 : 400}>{v?.toFixed(2)}</text>
          </g>
        )))}
      </svg>
      <div className="sq-heat-legend"><span>−1</span><div className="scale" /><span>+1</span></div>
    </div>
  )
}

function DistChart({ dist }) {
  const max = Math.max(...dist.real, ...dist.synthetic, 0.001)
  const isNum = dist.type === "numerical"
  const xs = isNum ? dist.bins : dist.categories
  // Y-axis ticks: 0, half, max (rounded)
  const top = Math.ceil(max)
  const yTicks = [top, Math.round(top / 2), 0]
  const fmtX = (v) => typeof v === "number" ? (Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 10) / 10) : v
  // show ~6 x labels evenly so they don't overlap
  const step = Math.max(1, Math.ceil(xs.length / 6))
  return (
    <div className="sq-chart">
      <div className="ch-top">
        <span className="ch-name">{dist.column}</span>
        <span className="ch-mu">{isNum ? `mean ${dist.real_mean} → ${dist.synth_mean}` : "label aligned"}</span>
      </div>
      <div className="sq-chart-body">
        <div className="sq-yaxis">
          {yTicks.map((t, i) => <span key={i}>{t}%</span>)}
        </div>
        <div className="sq-bars-wrap">
          <div className="sq-bars">
            {dist.real.map((r, i) => (
              <div key={i} className="sq-bcol" title={`${dist.column} ${isNum ? `≈ ${fmtX(xs[i])}` : xs[i]}\nReal: ${r}%\nSynthetic: ${dist.synthetic[i]}%`}>
                <div className="b real" style={{ height: `${(r / max) * 100}%` }} />
                <div className="b synth" style={{ height: `${(dist.synthetic[i] / max) * 100}%` }} />
              </div>
            ))}
          </div>
          <div className="sq-xaxis">
            {xs.map((x, i) => <span key={i} style={{ visibility: i % step === 0 || i === xs.length - 1 ? "visible" : "hidden" }}>{fmtX(x)}</span>)}
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState(null)
  const [summary, setSummary] = useState(null)
  const [issues, setIssues] = useState([])
  const [fixes, setFixes] = useState([])
  const [step, setStep] = useState("upload")
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [stage, setStage] = useState(0)
  const [error, setError] = useState(null)
  const [numRows, setNumRows] = useState(500)
  const [result, setResult] = useState(null)
  const [distributions, setDistributions] = useState(null)
  const [showDist, setShowDist] = useState(true)
  const [expandedIssue, setExpandedIssue] = useState(null)
  const [classRatios, setClassRatios] = useState({})
  const [history, setHistory] = useState([])
  const [targetColumn, setTargetColumn] = useState("")
  const [mode, setMode] = useState("tabular")
  const [textInfo, setTextInfo] = useState(null)
  const [textStrength, setTextStrength] = useState("medium")
  const [textLabel, setTextLabel] = useState("")
  const [textResult, setTextResult] = useState(null)
  const [showPro, setShowPro] = useState(false)
  const [stats, setStats] = useState(null)
  const [wlEmail, setWlEmail] = useState("")
  const [wlInterest, setWlInterest] = useState("")
  const [wlState, setWlState] = useState("idle") // idle | sending | done | error
  const [wlMsg, setWlMsg] = useState("")
  const [quip, setQuip] = useState(0)
  const [showDonate, setShowDonate] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    const id = setInterval(() => setQuip((q) => (q + 1) % FOOTER_QUIPS.length), 4000)
    return () => clearInterval(id)
  }, [])

  const fetchStats = async () => {
    try {
      const r = await fetch("http://localhost:8000/stats")
      if (r.ok) setStats(await r.json())
    } catch { /* stats are non-critical */ }
  }
  useEffect(() => { fetchStats() }, [])

  const submitWaitlist = async () => {
    setWlState("sending"); setWlMsg("")
    try {
      const r = await fetch("http://localhost:8000/waitlist", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: wlEmail || null, interest: wlInterest || null })
      })
      const d = await r.json()
      if (!d.ok) { setWlState("error"); setWlMsg(d.error || "Something went wrong."); return }
      setWlState("done")
      setWlMsg(d.already ? "You're already on the list — we'll be in touch!" : "You're on the list! We'll email you when Pro opens.")
      if (d.waitlist_count != null) setStats((s) => ({ ...(s || {}), waitlist_count: d.waitlist_count, waitlist_visible: d.waitlist_visible }))
    } catch {
      setWlState("error"); setWlMsg("Couldn't reach the server. Please try again.")
    }
  }
  const fmt = (n) => (n ?? 0).toLocaleString()

  const resetAll = () => {
    setFile(null); setSummary(null); setIssues([]); setFixes([]); setResult(null)
    setDistributions(null); setClassRatios({}); setTargetColumn(""); setTextInfo(null)
    setTextLabel(""); setTextResult(null); setMode("tabular"); setStep("upload"); setError(null)
  }

  const handleFile = (f) => {
    if (!f) return
    if (!f.name.endsWith(".csv")) { setError("Only CSV files are supported."); return }
    setFile(f); setError(null); setSummary(null); setIssues([]); setFixes([])
    setResult(null); setTextResult(null); setTextInfo(null); setStep("upload")
  }

  const handleAnalyse = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const tf = new FormData(); tf.append("file", file)
      const tr = await fetch("http://localhost:8000/analyse-text", { method: "POST", body: tf })
      if (!tr.ok) { const e = await tr.json().catch(() => ({})); throw new Error(e.detail || "Could not analyse this file.") }
      const td = await tr.json(); setTextInfo(td)
      const af = new FormData(); af.append("file", file)
      const r = await fetch("http://localhost:8000/analyse", { method: "POST", body: af })
      if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || "Could not analyse this file.") }
      const d = await r.json(); setSummary(d)
      setIssues(d.issues)
      setFixes(d.issues.map((i) => ({ column: i.column, issue: i.issue, fix_type: i.fix_type, approved: true })))
      setTargetColumn(d.suggested_target || "")
      if (td.is_text_dataset) { setMode("text"); setTextLabel(d.suggested_target || "") }
      else setMode("tabular")
      setStep("review")
    } catch (err) {
      const networkDown = (err instanceof TypeError) || /Failed to fetch|NetworkError|load failed/i.test(err?.message || "")
      setError(networkDown
        ? "Couldn't reach the backend. Make sure the server is running, then try again."
        : err.message)
    } finally { setLoading(false) }
  }

  const toggleFix = (i) => setFixes((p) => p.map((f, idx) => idx === i ? { ...f, approved: !f.approved } : f))

  const runStages = () => {
    const t = [800, 1500, 4000, 3000, 2000]; let c = 0
    t.forEach((d, i) => { c += d; setTimeout(() => setStage(i + 1), c) })
  }

  const handleGenerate = async () => {
    if (!file) return
    setGenerating(true); setError(null); setStage(0); runStages()
    const fd = new FormData(); fd.append("file", file); fd.append("fixes", JSON.stringify(fixes))
    const ar = Object.fromEntries(Object.entries(classRatios).filter(([_, v]) => v !== undefined && v > 0))
    const rp = Object.keys(ar).length ? `&class_ratios=${encodeURIComponent(JSON.stringify(ar))}` : ""
    const tp = targetColumn ? `&target_column=${encodeURIComponent(targetColumn)}` : ""
    try {
      const res = await fetch(`http://localhost:8000/generate-with-score?num_rows=${numRows}${rp}${tp}`, { method: "POST", body: fd })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail) }
      const d = await res.json(); setResult(d); setStep("result")
      setHistory((p) => [{ id: Date.now(), filename: file.name, rows: d.rows_generated, score: d.realism_score, kind: "tabular", csv: d.csv_data, time: new Date().toLocaleTimeString() }, ...p].slice(0, 5))
      if (d.distributions) setDistributions(d.distributions)
      fetchStats()
    } catch (e) {
      setError(e.message?.includes("500") ? "Generation failed — your dataset may have unsupported column types. Try approving more fixes." : e.message || "Something went wrong. Please try again.")
    } finally { setGenerating(false); setStage(0) }
  }

  const handleGenerateText = async () => {
    if (!file) return
    setGenerating(true); setError(null)
    const fd = new FormData(); fd.append("file", file)
    const lp = textLabel ? `&label_column=${encodeURIComponent(textLabel)}` : ""
    try {
      const res = await fetch(`http://localhost:8000/generate-text-hybrid?num_rows=${numRows}&augmentation_strength=${textStrength}${lp}`, { method: "POST", body: fd })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail) }
      const d = await res.json(); setTextResult(d); setStep("result")
      setHistory((p) => [{ id: Date.now(), filename: file.name, rows: d.generated_rows, score: d.quality.overall_score, kind: "text", csv: d.csv_data, time: new Date().toLocaleTimeString() }, ...p].slice(0, 5))
      fetchStats()
    } catch (e) { setError(e.message || "Text generation failed. Please try again.") } finally { setGenerating(false) }
  }

  const download = (csv, name) => {
    const blob = new Blob([csv], { type: "text/csv" }); const url = URL.createObjectURL(blob)
    const a = document.createElement("a"); a.href = url; a.download = `synthetic_${name}`; a.click(); URL.revokeObjectURL(url)
  }

  const approvedCount = fixes.filter(f => f.approved).length
  const activeCsv = mode === "text" ? textResult?.csv_data : result?.csv_data
  const preview = activeCsv ? (() => {
    const lines = activeCsv.trim().split("\n")
    return { headers: lines[0].split(","), rows: lines.slice(1, 8).map(l => l.split(",")) }
  })() : null
  const interp = result ? SCORE_INTERPRETATION(result.realism_score, result, summary) : null

  const steps = ["Upload", "Review", "Result"]
  const stepIndex = step === "upload" ? 0 : step === "review" ? 1 : 2

  return (
    <main>
      <header className="sq-header">
        <div className="sq-bar">
          <div className="sq-logo">
            <div className="sq-logo-mark">
              <svg viewBox="0 0 52 52" width="100%" height="100%">
                <rect x="14" y="16" width="12" height="5" rx="2.5" fill="#ffffff" />
                <rect x="14" y="24" width="19" height="5" rx="2.5" fill="#ffffff" />
                <rect x="14" y="32" width="26" height="5" rx="2.5" fill="#ffd400" />
              </svg>
            </div>
            <div className="sq-logo-word">Synthetic<b>Rows</b></div>
          </div>
          <div className="sq-steps">
            {steps.map((s, i) => {
              const done = i < stepIndex
              const isActive = i === stepIndex
              return (
                <div key={s} className={`sq-step ${isActive ? "is-active" : done ? "is-done" : ""}`}>
                  <div className={`sq-dot ${done ? "done" : isActive ? "active" : "todo"}`}>{done ? "✓" : i + 1}</div>
                  <span className="lbl">{s}</span>
                  {i < steps.length - 1 && <span className="sq-chev">›</span>}
                </div>
              )
            })}
          </div>
        </div>
      </header>

      <div className="sq-wrap">

        {/* ── SIDEBAR ── */}
        <aside className="sq-aside">
          {step === "upload" ? (
            <div className="sq-drop"
              onClick={() => !generating && inputRef.current.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}>
              <div className="ic">📄</div>
              {file ? <div className="h">{file.name}</div> : <><div className="h">Drop your CSV here</div><div className="s">or click to browse</div></>}
              <input ref={inputRef} type="file" accept=".csv" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
            </div>
          ) : (
            <div className="sq-crumb">
              <div className="ic">📄</div>
              <div className="body"><div className="ttl">{summary?.filename}</div><div className="sub">{summary?.rows} rows · {summary?.columns} columns</div></div>
              <button className="edit" onClick={resetAll} disabled={generating} style={generating ? { opacity: 0.4, cursor: "not-allowed" } : {}}>Change</button>
            </div>
          )}

          {error && <div className="sq-card pad" style={{ background: "var(--bad-wash)", borderColor: "#f0caca" }}><p style={{ color: "var(--bad)", fontSize: 13 }}>{error}</p></div>}

          {file && step === "upload" && <button className="sq-btn sq-btn-primary" onClick={handleAnalyse} disabled={loading}>{loading ? "Analysing…" : "Analyse dataset"}</button>}

          {summary && step !== "upload" && (
            <div className="sq-note">
              <div className="ic">◆</div>
              <div className="txt">
                {mode === "tabular"
                  ? <><b>Tabular dataset{textInfo?.is_text_dataset ? "" : " detected"}.</b> {step === "result" ? "We generated synthetic rows that preserve your column distributions and relationships." : "We'll generate synthetic rows that preserve your column distributions and relationships."}{textInfo?.is_text_dataset && !generating && <> <button className="switch" onClick={() => { setMode("text"); setTextLabel(summary.suggested_target || "") }}>Treat as text instead</button></>}</>
                  : <><b>Text dataset detected.</b> {step === "result" ? "We created new variations of your text columns while keeping every label aligned." : "We'll create new variations of your text columns while keeping every label correctly aligned."} {!generating && <button className="switch" onClick={() => setMode("tabular")}>Treat as tabular instead</button>}</>}
              </div>
            </div>
          )}

          {step === "result" && mode === "tabular" && result && (
            <div className="sq-card pad">
              <div className="sq-eyebrow" style={{ marginBottom: 10 }}>Run summary</div>
              {[["Rows generated", result.rows_generated], ["Fixes applied", approvedCount], ["Target column", targetColumn || "None"], ["Model", result.model_used]].map(([k, v]) => (
                <div className="sq-stat" key={k}><span className="k">{k}</span><span className="v mono">{v}</span></div>
              ))}
            </div>
          )}
          {step === "result" && mode === "text" && textResult && (
            <div className="sq-card pad">
              <div className="sq-eyebrow" style={{ marginBottom: 10 }}>Run summary</div>
              {[["Original rows", textResult.original_rows], ["Generated rows", textResult.generated_rows], ["Label column", textLabel || "None"], ["Method", textResult.model_used]].map(([k, v]) => (
                <div className="sq-stat" key={k}><span className="k">{k}</span><span className="v mono">{v}</span></div>
              ))}
            </div>
          )}

          {step === "review" && mode === "tabular" && issues.length > 0 && (
            <div className="sq-card pad">
              <div className="sq-eyebrow" style={{ marginBottom: 8 }}>Data quality · {issues.length} issue{issues.length !== 1 ? "s" : ""}</div>
              <p className="sq-issues-intro">We scanned your dataset and found {issues.length} things that could lower the quality of your synthetic data. Each fix is <b>on by default</b> — keep it enabled to let SyntheticRows clean it, or toggle it off to leave that column untouched. Tap <b>Why this matters</b> on any issue for details.</p>
              <div className={`sq-issues ${generating ? "locked" : ""}`}>
                {issues.map((issue, i) => {
                  const ex = ISSUE_EXPLANATIONS[issue.fix_type]?.(issue.column); const open = expandedIssue === i
                  return (
                    <div key={i} className={`sq-issue ${fixes[i]?.approved ? "" : "off"}`}>
                      <div className="head">
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span className="nm">{issue.column}</span>
                            <span className={`sev ${issue.severity === "high" ? "high" : "med"}`}>{issue.severity}</span>
                          </div>
                          <p className="desc">{issue.issue}</p>
                          {ex && (<>
                            <button className="why" onClick={() => !generating && setExpandedIssue(open ? null : i)}>{open ? "▼ Hide" : "▶ Why this matters"}</button>
                            {open && <div className="sq-expl">
                              <div className="b1"><p className="et" style={{ color: "var(--ink)" }}>What we found</p><p className="ed">{ex.what}</p></div>
                              <div className="b2"><p className="et" style={{ color: "var(--bad)" }}>Impact if ignored</p><p className="ed">{ex.impact}</p></div>
                              <div className="b3"><p className="et" style={{ color: "var(--good)" }}>What we'll do</p><p className="ed">{ex.action}</p></div>
                            </div>}
                          </>)}
                        </div>
                        <button className={`sq-sw ${fixes[i]?.approved ? "on" : "off"}`} onClick={() => !generating && toggleFix(i)} disabled={generating} />
                      </div>
                    </div>
                  )
                })}
              </div>
              {generating && <p className="sq-lock-note">🔒 Settings locked while generating…</p>}
            </div>
          )}

          {step === "review" && mode === "text" && textInfo && (
            <div className="sq-card pad">
              <div className="sq-eyebrow" style={{ marginBottom: 11 }}>Detected text columns</div>
              {textInfo.text_columns?.map((col) => (
                <div className="sq-tcol" key={col}>
                  <span className="tag">{col}</span>
                  {textInfo.previews?.[col]?.[0] && <div className="ex">"{textInfo.previews[col][0]}"</div>}
                </div>
              ))}
              {textInfo.non_text_columns?.length > 0 && <p className="sq-preserved">Kept and aligned: {textInfo.non_text_columns.join(", ")}</p>}
            </div>
          )}

          {history.length > 0 && (
            <div className="sq-card pad">
              <div className="sq-eyebrow" style={{ marginBottom: 8 }}>Session history</div>
              {history.map((h) => (
                <div className="sq-hist" key={h.id}>
                  <span className="nm">{h.filename}</span>
                  <span className="sc" style={{ color: sColor(h.score) }}>{h.score}</span>
                  <button className="dl" onClick={() => download(h.csv, h.filename)}>⬇</button>
                </div>
              ))}
            </div>
          )}
        </aside>

        {/* ── MAIN ── */}
        <section className="sq-main">

          {step === "upload" && (
            <div className="sq-empty"><div><div className="ic">📊</div><div className="h">Your synthetic data results will appear here</div><div className="s">Upload a CSV to begin</div></div></div>
          )}

          {step === "review" && mode === "tabular" && (
            <div className="sq-card pad-lg">
              <div className="sq-section" style={{ marginBottom: 4 }}>Generation settings</div>
              <p className="sq-intro" style={{ marginBottom: 20 }}>Configure how SyntheticRows generates your synthetic dataset. Defaults work well for most cases.</p>

              {summary?.is_imbalanced && (
                <div className="sq-banner"><p>Imbalanced classes ({Math.round(summary.imbalance_ratio * 100)}% / {Math.round((1 - summary.imbalance_ratio) * 100)}%) — preserved automatically.</p></div>
              )}

              <div className="sq-field">
                <div className="lab">Target column</div>
                <div className="help">The column your model predicts. Used for ML-readiness scoring and class balancing. Leave as None for unsupervised data.</div>
                <select className="sq-input" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} disabled={generating}>
                  <option value="">None (unsupervised)</option>
                  {summary?.column_names?.map((c) => <option key={c} value={c}>{c}{c === summary?.suggested_target ? " ✦ suggested" : ""}</option>)}
                </select>
                {summary?.suggested_target && (
                  <p className="sq-detected">SyntheticRows detected <b>{summary.suggested_target}</b> as the target column. If that's not the column your model predicts, pick the correct one above.</p>
                )}
              </div>

              <div className="sq-field">
                <div className="lab">Rows to generate · <span style={{ color: "var(--yellow-deep)" }}>{numRows}</span></div>
                <div className="help">How many synthetic rows to create. Free tier supports up to 1,000.</div>
                <input type="range" min="100" max="1000" step="50" value={numRows} onChange={(e) => setNumRows(+e.target.value)} disabled={generating} style={sliderFill(numRows)} />
                <div className="sq-slider-row"><span>100</span><span>1,000</span></div>
              </div>

              {summary?.target_column && summary?.target_classes?.length > 0 && (
                <div className="sq-field">
                  <div className="sq-class-head">
                    <div className="lab" style={{ margin: 0 }}>Class distribution</div>
                    <button type="button" className="sq-balance-btn" disabled={generating}
                      onClick={() => {
                        const classes = summary.target_classes
                        const k = classes.length
                        const base = Math.floor(numRows / k)
                        const rem = numRows - base * k
                        const next = {}
                        classes.forEach((c, i) => { next[c.value] = base + (i < rem ? 1 : 0) })
                        setClassRatios(next)
                      }}>
                      ⚖ Balance classes
                    </button>
                  </div>
                  <div className="help">Set custom row counts per class, or leave blank to keep the natural distribution.</div>
                  <div className="sq-class-grid">
                    {summary.target_classes.map((c) => (
                      <div key={c.value} className="sq-class-cell">
                        <label className="sq-class-label">{c.value} <span className="sq-class-count">· {c.count} real</span></label>
                        <input className="sq-input" type="number" min="0" placeholder="auto" value={classRatios[c.value] ?? ""} disabled={generating}
                          onChange={(e) => { const v = e.target.value; setClassRatios((p) => ({ ...p, [c.value]: v === "" ? undefined : parseInt(v) })) }} />
                      </div>
                    ))}
                  </div>
                  {(() => {
                    // Warn about classes that are too small to honor a requested count.
                    const tooSmall = summary.target_classes.filter(
                      (c) => typeof classRatios[c.value] === "number" && classRatios[c.value] > 0 && c.count < 10
                    )
                    if (tooSmall.length > 0) {
                      return (
                        <div className="sq-class-warn">
                          {tooSmall.map((c) => (
                            <div key={c.value}>⚠ <b>{c.value}</b> has only {c.count} real {c.count === 1 ? "example" : "examples"} — too few to generate reliably, so it may be left out of the result.</div>
                          ))}
                        </div>
                      )
                    }
                    return null
                  })()}
                  {(() => {
                    const counts = Object.values(classRatios).filter((v) => typeof v === "number" && v > 0)
                    const sum = counts.reduce((a, b) => a + b, 0)
                    if (counts.length > 0 && sum !== numRows) {
                      return (
                        <div className="sq-class-warn">
                          Your class counts add up to <b>{sum.toLocaleString()}</b>, so <b>{sum.toLocaleString()} rows</b> will be generated — the “rows to generate” total above will be ignored when custom class counts are set.
                        </div>
                      )
                    }
                    return null
                  })()}
                </div>
              )}

              {generating && (
                <div style={{ background: "#f7f7f4", borderRadius: 10, padding: 12, marginBottom: 18, display: "flex", flexDirection: "column", gap: 8 }}>
                  {GENERATION_STAGES.map((s) => {
                    const done = stage > s.id, active = stage === s.id
                    return (
                      <div key={s.id} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div className={active ? "animate-pulse" : ""} style={{ width: 16, height: 16, borderRadius: "50%", display: "grid", placeItems: "center", fontSize: 9, flex: "none", background: done ? "var(--good)" : active ? "var(--yellow)" : "var(--line-2)", color: active ? "var(--black)" : "#fff" }}>{done ? "✓" : active ? "●" : "○"}</div>
                        <span style={{ fontSize: 12, color: done ? "var(--good)" : active ? "var(--ink)" : "var(--faint)", fontWeight: active ? 600 : 400 }}>{s.label}</span>
                      </div>
                    )
                  })}
                </div>
              )}

              <button className="sq-btn sq-btn-primary" onClick={handleGenerate} disabled={generating}>{generating ? "Generating…" : "⚡ Generate synthetic dataset"}</button>
            </div>
          )}

          {step === "review" && mode === "text" && (
            <div className="sq-card pad-lg">
              <div className="sq-section" style={{ marginBottom: 4 }}>Augmentation settings</div>
              <p className="sq-intro" style={{ marginBottom: 20 }}>SyntheticRows rewrites your text into natural variations and pairs each one with a real, matching label so your dataset stays consistent.</p>

              <div className="sq-field">
                <div className="lab">Label column</div>
                <div className="help">The column that classifies each text row (e.g. sentiment). Augmented text stays matched to the correct label. Choose None if there's no label.</div>
                <select className="sq-input" value={textLabel} onChange={(e) => setTextLabel(e.target.value)} disabled={generating}>
                  <option value="">None</option>
                  {textInfo?.non_text_columns?.map((c) => <option key={c} value={c}>{c}{c === summary?.suggested_target ? " ✦ suggested" : ""}</option>)}
                </select>
              </div>

              <div className="sq-field">
                <div className="lab">Augmentation strength</div>
                <div className="help">How much each sentence changes. Medium gives natural variety without drifting from the original meaning.</div>
                <div className="sq-seg">
                  {["low", "medium", "high"].map((s) => <button key={s} className={textStrength === s ? "on" : ""} onClick={() => !generating && setTextStrength(s)} disabled={generating}>{s}</button>)}
                </div>
              </div>

              <div className="sq-field">
                <div className="lab">Rows to generate · <span style={{ color: "var(--yellow-deep)" }}>{numRows}</span></div>
                <div className="help">Total rows in the augmented dataset, including your originals.</div>
                <input type="range" min="100" max="1000" step="50" value={numRows} onChange={(e) => setNumRows(+e.target.value)} disabled={generating} style={sliderFill(numRows)} />
                <div className="sq-slider-row"><span>100</span><span>1,000</span></div>
              </div>

              {generating && <div style={{ background: "#f7f7f4", borderRadius: 10, padding: 12, marginBottom: 18, textAlign: "center" }}><p className="animate-pulse" style={{ color: "var(--yellow-deep)", fontSize: 13, fontWeight: 600 }}>Augmenting text and generating rows…</p></div>}
              <button className="sq-btn sq-btn-primary" onClick={handleGenerateText} disabled={generating}>{generating ? "Generating…" : "⚡ Generate augmented dataset"}</button>
            </div>
          )}

          {step === "result" && mode === "tabular" && result && (<>
            <div className="sq-card sq-hero">
              <div className="score-side">
                <div className="sq-eyebrow">Realism score</div>
                <div className="sq-score-num" style={{ color: sColor(result.realism_score) }}>{result.realism_score}</div>
                <div className="sq-score-grade" style={{ background: sWash(result.realism_score), color: sColor(result.realism_score) }}>{result.grade}</div>
              </div>
              <div className="break-side">
                <div className="sq-eyebrow" style={{ marginBottom: 14 }}>Score breakdown</div>
                {[["Statistical similarity", result.statistical_score, "50%"], ["Coverage", result.coverage_score, "30%"], ["Distinguishability", result.distinguishability_score, "20%"]].map(([l, v, w]) => (
                  <div className="sq-metric" key={l}>
                    <div className="top"><span className="name">{l}<span className="w">{w}</span></span><span className="val" style={{ color: sColor(v) }}>{v}</span></div>
                    <div className="sq-track"><div className="fill" style={{ width: `${v}%`, background: sColor(v) }} /></div>
                  </div>
                ))}
              </div>
            </div>

            {interp && (
              <div className="sq-card sq-verdict">
                <p>{interp.overall}</p>
                {interp.weakest && <p className="hint">💡 {interp.weakest}</p>}
              </div>
            )}

            {result.target_note && (
              <div className="sq-card sq-target-note">
                <span className="ic">◆</span>
                <p>{result.target_note}</p>
              </div>
            )}

            {result.skipped_classes && result.skipped_classes.length > 0 && (
              <div className="sq-card sq-skipped-note">
                <span className="ic">⚠</span>
                <div>
                  <b>We couldn't fully balance your data.</b>
                  {result.skipped_classes.map((s, i) => (
                    <p key={i}>
                      You asked for {s.requested.toLocaleString()} rows of class <b>{s.class_value}</b>, but your dataset only has <b>{s.real_examples} real {s.real_examples === 1 ? "example" : "examples"}</b> of it — too few to generate reliable synthetic rows. We left this class out rather than fabricate data from so few examples.
                    </p>
                  ))}
                  <p className="sq-skipped-advice">To balance this dataset, the most reliable fix is to collect more real examples of the rare class. With more minority data, we can generate a properly balanced set.</p>
                </div>
              </div>
            )}

            {result.tstr?.available && (
              <div className="sq-card pad-lg">
                <div className="sq-eyebrow" style={{ marginBottom: 10 }}>ML readiness — train on synthetic, test on real</div>
                <details className="sq-tstr-explain">
                  <summary>What is this, and how do I read it?</summary>
                  <div className="body">
                    <p>This is the real test of whether your synthetic data is good enough to <b>train a machine-learning model</b>. We train two models to predict your target column ({targetColumn || "the selected column"}): one on your <b>real</b> data, one on your <b>synthetic</b> data. Then we test <b>both on real data they've never seen</b>.</p>
                    <p><b>Real → Real</b> is the benchmark: how well a model does when trained on real data. <b>Synthetic → Real</b> is the one that matters: how well a model trained on your synthetic data performs on real data. The <b>accuracy gap</b> between them tells the story — a small gap means your synthetic data is nearly as useful as the real thing for training models.</p>
                  </div>
                </details>
                {(() => {
                  const isNeutral = result.tstr.color === "neutral"
                  const tColor = isNeutral ? "var(--muted)" : sColor(tstrScore(result.tstr.color))
                  const tWash = isNeutral ? "#f1f1ee" : sWash(tstrScore(result.tstr.color))
                  return (
                    <div className="sq-tstr">
                      <div className="sq-tstr-box"><div className="k">Real → Real</div><div className="v">{result.tstr.real_real_accuracy}%</div></div>
                      <div className="sq-tstr-box"><div className="k">Synthetic → Real</div><div className="v" style={{ color: tColor }}>{result.tstr.synth_real_accuracy}%</div></div>
                      <div className="sq-tstr-box flag" style={{ background: tWash }}><div className="k">Accuracy gap</div><div className="v" style={{ color: tColor }}>{result.tstr.performance_gap}%</div><div className="g" style={{ color: tColor }}>{result.tstr.grade}</div></div>
                    </div>
                  )
                })()}
                <p className="sq-intro">{result.tstr.interpretation}</p>
                {result.tstr.synth_real_accuracy <= 30 && (
                  <div className="sq-tstr-note">
                    <b>What a low score means.</b> A model trained on this synthetic data scored just {result.tstr.synth_real_accuracy}% on your real data — meaning the synthetic rows didn't preserve enough of the real signal between your features and the target ({targetColumn || "selected target"}) to train a useful model. This is an honest result, not an error. It usually happens when the dataset is small, noisy, heavily imbalanced, or the target is weakly related to the other columns. Try approving more data-quality fixes, picking a different target column, or generating more rows.
                  </div>
                )}
              </div>
            )}
            {!result.tstr?.available && result.tstr?.reason && (
              <div className="sq-card pad"><div className="sq-eyebrow" style={{ marginBottom: 4 }}>ML readiness</div><p style={{ fontSize: 12.5, color: "var(--faint)" }}>{result.tstr.reason}</p></div>
            )}

            <div className="sq-two">
              {result.correlations?.available && (
                <div className="sq-card pad-lg">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 15 }}>
                    <span className="sq-eyebrow">Correlation preservation</span>
                    <span className="sq-pill" style={{ background: result.correlations.avg_correlation_diff < 0.1 ? "var(--good-wash)" : result.correlations.avg_correlation_diff < 0.2 ? "var(--warn-wash)" : "var(--bad-wash)", color: result.correlations.avg_correlation_diff < 0.1 ? "var(--good)" : result.correlations.avg_correlation_diff < 0.2 ? "var(--warn)" : "var(--bad)" }}>diff {result.correlations.avg_correlation_diff}</span>
                  </div>
                  <div className="sq-heat-pair">
                    <Heatmap title="Real" columns={result.correlations.columns} matrix={result.correlations.real} />
                    <Heatmap title="Synthetic" columns={result.correlations.columns} matrix={result.correlations.synthetic} />
                  </div>
                </div>
              )}
              {result.column_quality?.length > 0 && (
                <div className="sq-card pad-lg">
                  <div className="sq-eyebrow" style={{ marginBottom: 15 }}>Column quality</div>
                  <div className="sq-colq">
                    {result.column_quality.map((c) => (
                      <div className="row" key={c.column}>
                        <span className="nm">{c.column}</span>
                        <span className="tr"><span className="fl" style={{ width: `${c.score}%`, background: sColor(c.score) }} /></span>
                        <span className="sc" style={{ color: sColor(c.score) }}>{c.score}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {distributions?.length > 0 && (
              <div className="sq-card pad-lg">
                <div className="sq-dist-head">
                  <span className="sq-eyebrow">Distribution comparison</span>
                  <button className="sq-toggle-link" onClick={() => setShowDist(!showDist)}>{showDist ? "Hide charts" : "Show charts"}</button>
                </div>
                {showDist && (<>
                  <div className="sq-dist-grid">{distributions.map((d) => <DistChart key={d.column} dist={d} />)}</div>
                  <div className="sq-legend">
                    <div className="it"><span className="sw" style={{ background: "var(--black)" }} />Real data</div>
                    <div className="it"><span className="sw" style={{ background: "var(--yellow)" }} />Synthetic data</div>
                  </div>
                </>)}
              </div>
            )}

            {preview && (
              <div className="sq-card pad-lg">
                <div className="sq-eyebrow" style={{ marginBottom: 14 }}>Synthetic data preview</div>
                <div className="sq-tbl">
                  <table>
                    <thead><tr>{preview.headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
                    <tbody>{preview.rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="mono">{c}</td>)}</tr>)}</tbody>
                  </table>
                </div>
              </div>
            )}

            {result.capped && <div className="sq-banner"><p>Capped at {result.max_recommended} rows (2× your real data) for quality.</p></div>}

            <button className="sq-pro-strip" onClick={() => setShowPro(true)}>
              <div className="sq-pro-strip-l">
                <span className="sq-pro-badge">PRO</span>
                <div>
                  <div className="t1">Want more from SyntheticRows?</div>
                  <div className="t2">10,000+ rows · AI-paraphrased text · image data · API access — see what Pro unlocks</div>
                </div>
              </div>
              <span className="sq-pro-cta">Compare plans →</span>
            </button>

            <div className="sq-actions">
              <button className="sq-btn sq-btn-primary" onClick={() => download(result.csv_data, file.name)}>⬇ Download synthetic dataset</button>
              <button className="sq-btn sq-btn-ghost" onClick={resetAll}>Start over</button>
            </div>
          </>)}

          {step === "result" && mode === "text" && textResult && (<>
            <div className="sq-three">
              <div className="sq-card pad-lg sq-stat-card">
                <div className="sq-eyebrow">Quality</div>
                <div style={{ fontSize: 48, fontWeight: 800, marginTop: 4, color: sColor(textResult.quality.overall_score) }}>{textResult.quality.overall_score}</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: sColor(textResult.quality.overall_score) }}>{textResult.quality.grade}</div>
              </div>
              <div className="sq-card pad-lg sq-stat-card">
                <div className="sq-eyebrow">Rows</div>
                <div style={{ fontSize: 28, fontWeight: 800, marginTop: 4 }}>{textResult.original_rows} → {textResult.generated_rows}</div>
                <div style={{ fontSize: 12, color: "var(--faint)" }}>original → generated</div>
              </div>
              <div className="sq-card pad-lg sq-stat-card">
                <div className="sq-eyebrow">Method</div>
                <div style={{ fontSize: 17, fontWeight: 700, marginTop: 4, color: "var(--yellow-deep)" }}>{textResult.model_used}</div>
                <div style={{ fontSize: 12, color: "var(--faint)" }}>labels stay aligned</div>
              </div>
            </div>

            <div className="sq-card pad-lg">
              <div className="sq-eyebrow" style={{ marginBottom: 15 }}>Text column quality</div>
              {Object.entries(textResult.quality.column_scores).map(([col, s]) => (
                <div className="sq-tq" key={col}>
                  <div className="tqtop"><span className="tqname">{col}</span><span className="tqsc" style={{ color: sColor(s.score) }}>{s.score}</span></div>
                  <div className="sq-track"><div className="fill" style={{ width: `${s.score}%`, background: sColor(s.score) }} /></div>
                  <div className="tqmeta"><span>Length kept: {s.length_preservation}%</span><span>Vocab diversity: {s.vocabulary_diversity}%</span></div>
                </div>
              ))}
            </div>

            {preview && (
              <div className="sq-card pad-lg">
                <div className="sq-eyebrow" style={{ marginBottom: 14 }}>Augmented data preview</div>
                <div className="sq-tbl">
                  <table>
                    <thead><tr>{preview.headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
                    <tbody>{preview.rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="wrap">{c}</td>)}</tr>)}</tbody>
                  </table>
                </div>
              </div>
            )}

            <button className="sq-pro-strip" onClick={() => setShowPro(true)}>
              <div className="sq-pro-strip-l">
                <span className="sq-pro-badge">PRO</span>
                <div>
                  <div className="t1">Want more from SyntheticRows?</div>
                  <div className="t2">10,000+ rows · AI-paraphrased text · image data · API access — see what Pro unlocks</div>
                </div>
              </div>
              <span className="sq-pro-cta">Compare plans →</span>
            </button>

            <div className="sq-actions">
              <button className="sq-btn sq-btn-primary" onClick={() => download(textResult.csv_data, file.name)}>⬇ Download augmented dataset</button>
              <button className="sq-btn sq-btn-ghost" onClick={resetAll}>Start over</button>
            </div>
          </>)}

        </section>
      </div>

      {/* ── Footer ── */}
      <footer className="sq-footer">
        <div className="sq-footer-inner">
          <div className="sq-footer-hero">
            <div className="sq-footer-stats">
              <div className="sq-footer-statbox">
                <div className="sq-footer-stat">{fmt(stats?.total_rows)}+</div>
                <div className="sq-footer-stat-label">synthetic rows generated</div>
              </div>
              <div className="sq-footer-statbox">
                <div className="sq-footer-stat">{fmt(stats?.total_datasets)}+</div>
                <div className="sq-footer-stat-label">datasets processed</div>
              </div>
            </div>
            <div className="sq-footer-quip" key={quip}>{FOOTER_QUIPS[quip]}</div>
          </div>
          <div className="sq-footer-grid">
            <div className="sq-footer-col sq-footer-brand">
              <div className="sq-footer-logo">Synthetic<b>Rows</b></div>
              <p>Turn small datasets into larger, realistic ones that train better models — with honest quality scores you can trust. Free to use, no signup required.</p>
            </div>
            <div className="sq-footer-col">
              <h5>Product</h5>
              <span>Tabular generation</span>
              <span>Text augmentation</span>
              <span>Realism scoring</span>
              <button className="sq-footer-link" onClick={() => setShowPro(true)}>Pro plan</button>
            </div>
            <div className="sq-footer-col">
              <h5>Why SyntheticRows</h5>
              <span>Honest 3-metric scoring</span>
              <span>ML-readiness (TSTR)</span>
              <span>Privacy-safe synthetic data</span>
              <span>Built for ML teams</span>
            </div>
            <div className="sq-footer-col">
              <h5>Get started</h5>
              <span>Upload a CSV</span>
              <span>Generate up to 1,000 rows free</span>
              <button className="sq-footer-link" onClick={() => setShowPro(true)}>Join the Pro waitlist</button>
              <a className="sq-footer-link" href="https://docs.google.com/forms/d/e/1FAIpQLScHyvSe14UAJeBMT3l5ZH7s182SG5MYgIVBsqnk4nIbXp_F-A/viewform" target="_blank" rel="noopener noreferrer">Share a suggestion</a>
            </div>
          </div>
          <div className="sq-footer-bottom">
            <span>© {new Date().getFullYear()} SyntheticRows. Built for people who care about data quality.</span>
            <span>Made with care in India 🇮🇳</span>
          </div>
        </div>
      </footer>

      {/* ── Pro comparison modal ── */}
      {showPro && (
        <div className="sq-modal-overlay" onClick={() => setShowPro(false)}>
          <div className="sq-modal" onClick={(e) => e.stopPropagation()}>
            <button className="sq-modal-close" onClick={() => setShowPro(false)}>✕</button>
            <div className="sq-modal-head">
              <div className="sq-eyebrow" style={{ color: "var(--yellow-deep)" }}>Upgrade</div>
              <h2 className="sq-modal-title">Do more with SyntheticRows Pro</h2>
              <p className="sq-modal-sub">You're using the free version. Here's what Pro unlocks — built for when prototypes become real models.</p>
            </div>

            <div className="sq-plans">
              <div className="sq-plan">
                <div className="sq-plan-name">Free</div>
                <div className="sq-plan-price">$0<span>/forever</span></div>
                <div className="sq-plan-tag">No signup required</div>
                <ul className="sq-plan-list">
                  <li>Generate up to <b>1,000</b> synthetic rows</li>
                  <li>Honest 3-metric realism score</li>
                  <li>Auto data-quality fixes, explained simply</li>
                  <li>Distribution &amp; correlation charts</li>
                  <li>ML-readiness (TSTR) scoring</li>
                  <li>Text augmentation (local synonym engine)</li>
                  <li>Instant CSV download</li>
                </ul>
                <div className="sq-plan-foot sq-plan-foot-current">Your current plan</div>
              </div>

              <div className="sq-plan pro">
                <div className="sq-plan-popular">MOST POPULAR</div>
                <div className="sq-plan-name">Pro</div>
                <div className="sq-plan-price">$X.XX<span>/month</span></div>
                <div className="sq-plan-tag">Starting at just $X.XX/month</div>
                <ul className="sq-plan-list">
                  <li className="sq-plan-everything">✦ Everything in Free, plus —</li>
                  <li><b>10,000+ rows per run</b> — train real models, not just prototypes</li>
                  <li><b>AI-paraphrased text</b> — human-quality variations, not word swaps</li>
                  <li><b>Image data augmentation</b> — grow image datasets, not just tables &amp; text</li>
                  <li><b>API access</b> — pipe SyntheticRows straight into your ML pipeline</li>
                  <li><b>Priority generation</b> — skip the queue, get results faster</li>
                  <li><b>Advanced domain controls</b> — enforce your own rules &amp; constraints</li>
                  <li><b>Shareable PDF reports</b> — prove data quality to your team or clients</li>
                  <li><b>Commercial license</b> + email support</li>
                </ul>
                <div className="sq-plan-foot">
                  {wlState === "done" ? (
                    <div className="sq-wl-done">✓ {wlMsg}</div>
                  ) : (
                    <div className="sq-wl">
                      <input className="sq-input" type="email" placeholder="you@email.com (optional)" value={wlEmail} onChange={(e) => setWlEmail(e.target.value)} />
                      <select className="sq-input" value={wlInterest} onChange={(e) => setWlInterest(e.target.value)}>
                        <option value="">What matters most to you?</option>
                        <option>Better text quality (LLM)</option>
                        <option>Larger datasets</option>
                        <option>Image data</option>
                        <option>API access</option>
                        <option>Just exploring</option>
                      </select>
                      <button className="sq-btn sq-btn-primary" onClick={submitWaitlist} disabled={wlState === "sending"}>
                        {wlState === "sending" ? "Joining…" : "⚡ Get early access"}
                      </button>
                      {wlState === "error" && <p className="sq-wl-err">{wlMsg}</p>}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {stats?.waitlist_visible
              ? <p className="sq-wl-count">Join <b>{fmt(stats.waitlist_count)}</b> others on the Pro waitlist</p>
              : <p className="sq-wl-count">Be one of the first to get Pro — early-access members get founding pricing.</p>}
          </div>
        </div>
      )}

      {/* ── Floating donation (pizza) button ── */}
      <button className="sq-pizza" onClick={() => setShowDonate(true)} title="Buy me a pizza" aria-label="Support SyntheticRows">
        <span className="sq-pizza-ic">🍕</span>
        <span className="sq-pizza-label">Buy me a pizza</span>
      </button>

      {showDonate && (
        <div className="sq-modal-overlay" onClick={() => setShowDonate(false)}>
          <div className="sq-donate" onClick={(e) => e.stopPropagation()}>
            <button className="sq-modal-close" onClick={() => setShowDonate(false)}>✕</button>
            <div className="sq-donate-ic">🍕</div>
            <h2 className="sq-donate-title">Fuel the next feature</h2>
            <p className="sq-donate-copy">SyntheticRows is free and built by one sleep-deprived student. A pizza goes further than you'd think — buy me one and I'll generate synthetic gratitude (realism score: 100). 🍕</p>
            <a className="sq-btn sq-btn-primary" href={DONATE_URL} target="_blank" rel="noopener noreferrer" onClick={() => setShowDonate(false)}>🍕 Buy me a pizza</a>
            <button className="sq-donate-skip" onClick={() => setShowDonate(false)}>Maybe later</button>
          </div>
        </div>
      )}
    </main>
  )
}