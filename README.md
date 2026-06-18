<div align="center">

# SyntheticRows

### Turn a small CSV into a bigger, realistic dataset — with quality scoring you can actually trust.

[**syntheticrows.com**](https://syntheticrows.com)

</div>

---

## The problem

Good machine-learning models need enough data — but real data is often scarce, imbalanced, or sensitive. Most "synthetic data" tools either generate fake-looking dummy records from scratch, hide their quality behind marketing, or lock everything behind enterprise pricing.

## What SyntheticRows does

SyntheticRows learns the real statistical patterns in **your own CSV** and generates additional rows that look and behave like the real thing — then tells you, honestly, how good the result actually is.

Upload a small dataset. Get back a larger one that preserves your data's distributions, correlations, and structure — along with a transparent quality score and a real test of whether the synthetic data is good enough to train a model on.

No signup. No black box. No inflated scores.

## Features

- **Synthetic tabular generation** — powered by CTGAN, TVAE, and GaussianCopula (via SDV), with automatic model selection based on your data.
- **Honest realism scoring** — a transparent three-part score covering distinguishability, statistical similarity, and coverage. When the data is poor, it says so.
- **ML-readiness testing (TSTR)** — trains a model on your synthetic data and evaluates it on real data, so you know if it's genuinely usable for training, not just statistically pretty.
- **Class imbalance handling** — rebalance class distributions on demand, with honest warnings when a minority class has too few real examples to synthesize reliably.
- **Automatic data-quality fixes** — detects and handles missing values, outliers, and degenerate columns before generation.
- **Text augmentation** — supports datasets containing free-text columns.

## Why it's different

Most tools optimize for data that *looks* real. SyntheticRows optimizes for data you can *trust* — and is honest when the data isn't good enough. That transparency is the whole point: you should never ship a model trained on synthetic data without knowing whether it actually holds up.

## Tech stack

| Layer | Technologies |
|-------|-------------|
| Backend | FastAPI · SDV (CTGAN / TVAE / GaussianCopula) · scikit-learn · pandas · PyTorch |
| Frontend | Next.js (App Router) · Tailwind CSS |
| Deployment | Hugging Face Spaces (backend) · Vercel (frontend) |

## Project structure

    synthiq/
    ├── backend/                FastAPI backend — the synthetic data engine
    │   ├── app/                Generation, scoring, profiling, validation
    │   ├── Dockerfile          Container setup for deployment
    │   └── requirements.txt    Python dependencies
    └── frontend/               Next.js frontend
        └── app/                Pages, components, and styles

## Running locally

**Backend**

    cd backend
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload

**Frontend**

    cd frontend
    npm install
    npm run dev

The frontend reads the backend URL from `NEXT_PUBLIC_API_URL`, defaulting to `http://localhost:8000` for local development.

---

<div align="center">

Built by [@dhruvgoyal16](https://github.com/dhruvgoyal16)

</div>
