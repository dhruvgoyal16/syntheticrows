from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import io
import json

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
        "has_datetime": profile.has_datetime,
        "columns_to_drop": profile.columns_to_drop,
        "issues": all_issues,
        "issues_found": len(all_issues)
    }


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
        result = validate(cleaned_df, synthetic_df)

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
            "csv_data": output.getvalue()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))