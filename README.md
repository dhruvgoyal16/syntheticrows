# SynthIQ 🧬

> Generate realistic synthetic datasets from your small real-world data — without losing statistical quality.

SynthIQ is an open-source synthetic data generation tool built for ML developers, researchers, and students who struggle with small datasets. Upload a CSV, get back a larger, realistic version that trains better models.

---

## Why SynthIQ?

Most synthetic data tools generate data that *looks* right but *trains wrong* — causing overfitting or underfitting. SynthIQ solves this by:

- **Automatically profiling your dataset** — detecting column types, missing values, zero inflation, outliers, and more
- **Letting you control fixes** — you decide what gets cleaned before generation, nothing is changed silently
- **Picking the right model** — GaussianCopula for small clean data, TVAE for tiny/noisy data, CTGAN for large complex data
- **Scoring the output honestly** — a three-metric Realism Score tells you exactly how good your synthetic data is

---

## Features

- Upload any CSV dataset
- Automatic data quality report with toggleable fixes
- Smart model selection based on dataset size and type
- Synthetic data generation using CTGAN, TVAE, and GaussianCopula
- Three-metric Realism Score (Distinguishability + Statistical Similarity + Coverage)
- Column-level quality report
- Download synthetic dataset as CSV

---

## Tech Stack

**Frontend** — Next.js, Tailwind CSS

**Backend** — FastAPI, Python

**Generation Engine** — SDV (CTGAN, TVAE, GaussianCopulaSynthesizer)

**Validation** — scikit-learn, pandas, numpy

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/synthiq.git
cd synthiq
```

### 2. Set up the backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at `http://localhost:8000`

### 3. Set up the frontend

Open a new terminal tab:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`

### 4. Open the app

Go to `http://localhost:3000` in your browser and upload a CSV file to get started.

---

## How It Works

1. **Upload CSV** — drag and drop your dataset
2. **Data Profiler** — detects column types, issues, and dataset type
3. **Smart Preprocessor** — applies only your approved fixes
4. **Model Router** — selects the optimal generation model automatically
5. **Generation Engine** — trains the model and generates synthetic rows
6. **Validator** — scores output across three quality metrics
7. **Download** — get your synthetic CSV with a full quality report

## Realism Score

The Realism Score is a weighted average of three metrics:

| Metric | Weight | What it measures |
|---|---|---|
| Statistical Similarity | 50% | Do distributions match column by column? |
| Coverage | 30% | Does synthetic data cover the full data range? |
| Distinguishability | 20% | Can a classifier tell real from synthetic? |

A score of **80+** is Excellent. **60-79** is Good. Below 60 is Fair.

---

## Supported Data Types

| Type | Support |
|---|---|
| Numerical (continuous) | ✅ Full |
| Numerical (discrete) | ✅ Full |
| Categorical (low cardinality) | ✅ Full |
| Boolean | ✅ Full |
| DateTime | ✅ Feature extraction |
| High cardinality categorical | ⚠️ Auto-dropped |
| ID columns | ⚠️ Auto-dropped |
| Time series | 🔜 Coming soon |
| Image datasets | 🔜 Coming soon |
| Text datasets | 🔜 Coming soon |

---

## Roadmap

- [x] Tabular data generation
- [x] Data quality report
- [x] Smart model selection
- [x] Realism Score
- [x] Column quality report
- [ ] User authentication
- [ ] Usage tiers (Free / Pro / Enterprise)
- [ ] Text dataset support
- [ ] Image dataset augmentation
- [ ] API access

---

## Contributing

This project is in active development. Contributions, issues, and feature requests are welcome.

---

## License

MIT

---

Built by Dhruv Goyal