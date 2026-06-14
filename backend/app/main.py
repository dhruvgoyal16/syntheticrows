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
from .text_augmentor import detect_text_columns, augment_dataset, score_text_augmentation

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

        real_pct = [0.0 if np.isnan(v) else round(float(v), 2) for v in (real_counts / len(real_vals) * 100)]
        synth_pct = [0.0 if np.isnan(v) else round(float(v), 2) for v in (synth_counts / len(synth_vals) * 100)]
        bin_labels = [round(float(b), 2) for b in bins[:-1]]

        def safe_float(v):
            return 0.0 if (v is None or np.isnan(v)) else round(float(v), 3)

        distributions.append({
            "column": col,
            "type": "numerical",
            "bins": bin_labels,
            "real": real_pct,
            "synthetic": synth_pct,
            "real_mean": safe_float(real_vals.mean()),
            "synth_mean": safe_float(synth_vals.mean()),
            "real_std": safe_float(real_vals.std()),
            "synth_std": safe_float(synth_vals.std()),
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
    real_numeric = real_df.select_dtypes(include=[np.number])
    synth_numeric = synthetic_df.select_dtypes(include=[np.number])

    common_cols = [c for c in real_numeric.columns if c in synth_numeric.columns]

    if len(common_cols) < 2:
        return {"available": False}

    real_corr = real_numeric[common_cols].corr().round(3)
    synth_corr = synth_numeric[common_cols].corr().round(3)

    # Replace NaN with 0
    real_corr = real_corr.fillna(0)
    synth_corr = synth_corr.fillna(0)

    diff_corr = (real_corr - synth_corr).abs().round(3)
    
    upper_vals = diff_corr.values[np.triu_indices_from(diff_corr.values, k=1)]
    upper_vals = upper_vals[~np.isnan(upper_vals)]
    avg_diff = round(float(upper_vals.mean()), 3) if len(upper_vals) > 0 else 0.0

    return {
        "available": True,
        "columns": common_cols,
        "real": [[0.0 if np.isnan(v) else round(float(v), 3) for v in row] for row in real_corr.values],
        "synthetic": [[0.0 if np.isnan(v) else round(float(v), 3) for v in row] for row in synth_corr.values],
        "diff": [[0.0 if np.isnan(v) else round(float(v), 3) for v in row] for row in diff_corr.values],
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
        "suggested_target": profile.target_column,  # suggestion only
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
    fixes: str = "[]",
    class_ratios: str = "{}",
    target_column: str = ""
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

    # Auto preprocess
    cleaned_df = auto_preprocess(cleaned_df, profile)

    try:
        # Parse custom class ratios if provided
        try:
            parsed_ratios = json.loads(class_ratios)
        except Exception:
            parsed_ratios = {}

        print("DEBUG class_ratios received:", class_ratios)
        print("DEBUG parsed_ratios:", parsed_ratios)

        # Override profile target with user confirmed target
        if target_column.strip():
            profile.target_column = target_column.strip()
            # Re-detect imbalance with confirmed target
            from .profiler import detect_imbalance
            is_imb, imb_ratio = detect_imbalance(cleaned_df, target_column.strip())
            profile.is_imbalanced = is_imb
            profile.imbalance_ratio = imb_ratio

        synthetic_df, model_used = generate(cleaned_df, profile, num_rows, parsed_ratios)
        # Use user-confirmed target if provided, else fall back to detected
        confirmed_target = target_column.strip() if target_column.strip() else profile.target_column
        result = validate(cleaned_df, synthetic_df, confirmed_target)

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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse-text")
async def analyse_text(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    text_cols = detect_text_columns(df)
    non_text_cols = [c for c in df.columns if c not in text_cols]

    # Sample texts for preview
    previews = {}
    for col in text_cols:
        sample = df[col].dropna().head(3).tolist()
        previews[col] = sample

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "text_columns": text_cols,
        "non_text_columns": non_text_cols,
        "text_column_count": len(text_cols),
        "previews": previews,
        "is_text_dataset": len(text_cols) > 0
    }


@app.post("/augment-text")
async def augment_text_endpoint(
    file: UploadFile = File(...),
    num_rows: int = 500,
    augmentation_strength: str = "medium"
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if num_rows > 1000:
        num_rows = 1000

    max_recommended = len(df) * 3
    capped = False
    if num_rows > max_recommended:
        num_rows = max_recommended
        capped = True

    text_cols = detect_text_columns(df)

    if not text_cols:
        raise HTTPException(
            status_code=400,
            detail="No text columns detected in this dataset. Use the tabular generation endpoint instead."
        )

    try:
        augmented_df = augment_dataset(
            df,
            text_cols,
            num_rows,
            augmentation_strength
        )

        quality = score_text_augmentation(df, augmented_df, text_cols)

        output = io.StringIO()
        augmented_df.to_csv(output, index=False)

        return {
            "original_rows": len(df),
            "augmented_rows": len(augmented_df),
            "rows_generated": len(augmented_df) - len(df),
            "text_columns": text_cols,
            "augmentation_strength": augmentation_strength,
            "capped": capped,
            "max_recommended": max_recommended,
            "quality": quality,
            "csv_data": output.getvalue()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-text-hybrid")
async def generate_text_hybrid(
    file: UploadFile = File(...),
    num_rows: int = 500,
    augmentation_strength: str = "medium",
    label_column: str = ""
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if num_rows > 1000:
        num_rows = 1000

    text_cols = detect_text_columns(df)

    if not text_cols:
        raise HTTPException(
            status_code=400,
            detail="No text columns detected. Use the tabular generation endpoint instead."
        )

    profile = profile_dataset(df, file.filename)

    # Determine label column — user-provided or detected
    confirmed_label = label_column.strip() if label_column.strip() else profile.target_column

    try:
        from .text_augmentor import hybrid_generate

        result_df, model_used = hybrid_generate(
            df,
            text_cols,
            num_rows,
            profile,
            augmentation_strength,
            confirmed_label
        )

        quality = score_text_augmentation(df, result_df, text_cols)

        output = io.StringIO()
        result_df.to_csv(output, index=False)

        return {
            "original_rows": len(df),
            "generated_rows": len(result_df),
            "text_columns": text_cols,
            "tabular_columns": [c for c in df.columns if c not in text_cols],
            "label_column": confirmed_label,
            "model_used": model_used,
            "augmentation_strength": augmentation_strength,
            "quality": quality,
            "csv_data": output.getvalue()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))