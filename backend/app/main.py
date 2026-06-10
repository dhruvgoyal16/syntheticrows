from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import io
import json
import numpy as np
from .models import Fix
from .profiler import profile_dataset
from .preprocessor import apply_fixes, auto_preprocess
from .generator import generate
from .validator import validate

app = FastAPI(
    title="SynthIQ API",
    description="Synthetic data generation engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, num_bins: int = 20) -> list:
    distributions = []

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns
    categorical_cols = real_df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if col not in synthetic_df.columns:
            continue

        real_vals = real_df[col].dropna()
        synth_vals = synthetic_df[col].dropna()
         # Treat binary columns as categorical
        if real_vals.nunique() == 2:
            categories = sorted(real_vals.unique().tolist())
            real_counts = real_vals.value_counts(normalize=True) * 100
            synth_counts = synth_vals.value_counts(normalize=True) * 100
            distributions.append({
                "column": col,
                "type": "categorical",
                "categories": [str(c) for c in categories],
                "real": [round(float(real_counts.get(c, 0)), 2) for c in categories],
                "synthetic": [round(float(synth_counts.get(c, 0)), 2) for c in categories],
            })
            continue

        combined_min = min(real_vals.min(), synth_vals.min())
        combined_max = max(real_vals.max(), synth_vals.max())

        if combined_min == combined_max:
            continue

        bins = np.linspace(combined_min, combined_max, num_bins + 1)
        real_counts, _ = np.histogram(real_vals, bins=bins)
        synth_counts, _ = np.histogram(synth_vals, bins=bins)

        real_pct = (real_counts / len(real_vals) * 100).round(2).tolist()
        synth_pct = (synth_counts / len(synth_vals) * 100).round(2).tolist()
        bin_labels = [round(float(b), 2) for b in bins[:-1]]

        distributions.append({
            "column": col,
            "type": "numerical",
            "bins": bin_labels,
            "real": real_pct,
            "synthetic": synth_pct,
            "real_mean": round(float(real_vals.mean()), 3),
            "synth_mean": round(float(synth_vals.mean()), 3),
            "real_std": round(float(real_vals.std()), 3),
            "synth_std": round(float(synth_vals.std()), 3),
        })

    for col in categorical_cols:
        if col not in synthetic_df.columns:
            continue

        real_counts = real_df[col].value_counts(normalize=True) * 100
        synth_counts = synthetic_df[col].value_counts(normalize=True) * 100
        categories = real_counts.index.tolist()

        distributions.append({
            "column": col,
            "type": "categorical",
            "categories": categories,
            "real": [round(float(real_counts.get(c, 0)), 2) for c in categories],
            "synthetic": [round(float(synth_counts.get(c, 0)), 2) for c in categories],
        })

    return distributions

def compute_correlations(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
    """Compute correlation matrices for real and synthetic data."""
    real_numeric = real_df.select_dtypes(include=[np.number])
    synth_numeric = synthetic_df.select_dtypes(include=[np.number])

    common_cols = [c for c in real_numeric.columns if c in synth_numeric.columns]

    if len(common_cols) < 2:
        return {"available": False}

    real_corr = real_numeric[common_cols].corr().round(3)
    synth_corr = synth_numeric[common_cols].corr().round(3)

    # Compute difference matrix
    diff_corr = (real_corr - synth_corr).abs().round(3)
    avg_diff = round(float(diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)].mean()), 3)

    return {
        "available": True,
        "columns": common_cols,
        "real": real_corr.values.tolist(),
        "synthetic": synth_corr.values.tolist(),
        "diff": diff_corr.values.tolist(),
        "avg_correlation_diff": avg_diff,
        "interpretation": (
            "Excellent — inter-column relationships are very well preserved." if avg_diff < 0.1 else
            "Good — most inter-column relationships are preserved." if avg_diff < 0.2 else
            "Fair — some inter-column relationships differ from real data." if avg_diff < 0.3 else
            "Poor — significant correlation differences detected. Consider approving more fixes."
        )
    }

@app.get("/")
def root():
    return {"message": "SynthIQ backend is running", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse")
async def analyse_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    profile = profile_dataset(df, file.filename)

    # Flatten issues for frontend
    all_issues = []
    for col in profile.columns:
        for issue in col.issues:
            all_issues.append({
                "column": col.name,
                "issue": issue.issue,
                "fix_type": issue.fix_type,
                "recommendation": issue.recommendation,
                "severity": issue.severity
            })

    return {
        "filename": profile.filename,
        "rows": profile.num_rows,
        "columns": profile.num_cols,
        "column_names": [c.name for c in profile.columns],
        "size_category": profile.size_category,
        "dataset_type": profile.dataset_type,
        "is_imbalanced": profile.is_imbalanced,
        "imbalance_ratio": profile.imbalance_ratio,
        "has_datetime": profile.has_datetime,
        "target_column": profile.target_column,
        "columns_to_drop": profile.columns_to_drop,
        "issues": all_issues,
        "issues_found": len(all_issues)
    }

@app.post("/distributions")
async def get_distributions(
    file: UploadFile = File(...),
    synthetic_csv: str = "",
    num_bins: int = 20
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    real_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if not synthetic_csv:
        raise HTTPException(status_code=400, detail="Synthetic CSV data is required")

    synthetic_df = pd.read_csv(io.StringIO(synthetic_csv))

    distributions = []

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns
    categorical_cols = real_df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if col not in synthetic_df.columns:
            continue

        real_vals = real_df[col].dropna()
        synth_vals = synthetic_df[col].dropna()

        # Compute shared bin edges
        combined_min = min(real_vals.min(), synth_vals.min())
        combined_max = max(real_vals.max(), synth_vals.max())

        if combined_min == combined_max:
            continue

        bins = np.linspace(combined_min, combined_max, num_bins + 1)

        real_counts, _ = np.histogram(real_vals, bins=bins)
        synth_counts, _ = np.histogram(synth_vals, bins=bins)

        # Normalize to percentages
        real_pct = (real_counts / len(real_vals) * 100).round(2).tolist()
        synth_pct = (synth_counts / len(synth_vals) * 100).round(2).tolist()

        bin_labels = [round(float(b), 2) for b in bins[:-1]]

        distributions.append({
            "column": col,
            "type": "numerical",
            "bins": bin_labels,
            "real": real_pct,
            "synthetic": synth_pct,
            "real_mean": round(float(real_vals.mean()), 3),
            "synth_mean": round(float(synth_vals.mean()), 3),
            "real_std": round(float(real_vals.std()), 3),
            "synth_std": round(float(synth_vals.std()), 3),
        })

    for col in categorical_cols:
        if col not in synthetic_df.columns:
            continue

        real_counts = real_df[col].value_counts(normalize=True) * 100
        synth_counts = synthetic_df[col].value_counts(normalize=True) * 100

        categories = real_counts.index.tolist()

        distributions.append({
            "column": col,
            "type": "categorical",
            "categories": categories,
            "real": [round(float(real_counts.get(c, 0)), 2) for c in categories],
            "synthetic": [round(float(synth_counts.get(c, 0)), 2) for c in categories],
        })

    return {"distributions": distributions}

@app.post("/generate-with-score")
async def generate_with_score(
    file: UploadFile = File(...),
    num_rows: int = 500,
    fixes: str = "[]"
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    real_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Cap rows
    if num_rows > 1000:
        num_rows = 1000

    max_recommended = len(real_df) * 2
    capped = False
    if num_rows > max_recommended:
        num_rows = max_recommended
        capped = True

    # Profile the dataset
    profile = profile_dataset(real_df, file.filename)

    # Apply user approved fixes
    try:
        fixes_list = [Fix(**f) for f in json.loads(fixes)]
        cleaned_df = apply_fixes(real_df, fixes_list)
    except Exception:
        cleaned_df = real_df

    # Auto preprocess (drop IDs, handle datetime, rare categories)
    cleaned_df = auto_preprocess(cleaned_df, profile)

    try:
        synthetic_df, model_used = generate(cleaned_df, profile, num_rows)
        result = validate(cleaned_df, synthetic_df, profile.target_column)

        output = io.StringIO()
        synthetic_df.to_csv(output, index=False)

        return {
            "realism_score": result.final_score,
            "distinguishability_score": result.distinguishability_score,
            "statistical_score": result.statistical_score,
            "coverage_score": result.coverage_score,
            "grade": result.grade,
            "color": result.color,
            "rows_generated": len(synthetic_df),
            "model_used": model_used,
            "capped": capped,
            "max_recommended": max_recommended,
            "column_quality": [cq.dict() for cq in result.column_quality],
            "tstr": result.tstr,
            "distributions": compute_distributions(cleaned_df, synthetic_df),
            "correlations": compute_correlations(cleaned_df, synthetic_df),
            "csv_data": output.getvalue()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))