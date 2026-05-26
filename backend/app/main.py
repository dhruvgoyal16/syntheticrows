from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import io
import json

app = FastAPI(
    title="SynthIQ API",
    description="Synthetic data generation engine",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────────────────────────────

class Fix(BaseModel):
    column: str
    issue: str
    fix_type: str
    approved: bool

# ─── Helpers ──────────────────────────────────────────────────────────────────

def detect_issues(df: pd.DataFrame) -> List[dict]:
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        total = len(df[col])

        # 1. Missing values (actual NaN)
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            pct = round((missing_count / total) * 100, 1)
            issues.append({
                "column": col,
                "issue": f"{missing_count} missing values ({pct}% of rows)",
                "fix_type": "fill_missing",
                "recommendation": f"Fill with column median",
                "severity": "high" if pct > 20 else "medium"
            })

        # 2. Zero inflation
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / total) * 100
        if zero_pct > 10 and col.lower() not in ["outcome", "target", "label", "class"]:
            issues.append({
                "column": col,
                "issue": f"{zero_count} zero values ({round(zero_pct, 1)}% of rows) — may indicate missing data",
                "fix_type": "fix_zeros",
                "recommendation": f"Replace zeros with column median ({round(df[col].replace(0, np.nan).median(), 2)})",
                "severity": "high" if zero_pct > 30 else "medium"
            })

        # 3. Outliers
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            outlier_count = ((df[col] - mean).abs() > 3 * std).sum()
            if outlier_count > 0:
                issues.append({
                    "column": col,
                    "issue": f"{outlier_count} outliers detected (beyond 3 std deviations)",
                    "fix_type": "cap_outliers",
                    "recommendation": f"Cap values at ±3 standard deviations",
                    "severity": "medium"
                })

        # 4. Constant columns
        if df[col].nunique() == 1:
            issues.append({
                "column": col,
                "issue": f"Column has only one unique value — useless for training",
                "fix_type": "drop_column",
                "recommendation": "Drop this column before generation",
                "severity": "high"
            })

    return issues


def apply_fixes(df: pd.DataFrame, fixes: List[Fix]) -> pd.DataFrame:
    df = df.copy()

    for fix in fixes:
        if not fix.approved:
            continue

        col = fix.column
        if col not in df.columns:
            continue

        if fix.fix_type == "fill_missing":
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        elif fix.fix_type == "fix_zeros":
            median_val = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, median_val)

        elif fix.fix_type == "cap_outliers":
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)

        elif fix.fix_type == "drop_column":
            df = df.drop(columns=[col])

    return df


def calculate_realism_score(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
    real_numeric = real_df.select_dtypes(include=[np.number])
    synthetic_numeric = synthetic_df.select_dtypes(include=[np.number])

    common_cols = list(set(real_numeric.columns) & set(synthetic_numeric.columns))
    real_numeric = real_numeric[common_cols].fillna(0)
    synthetic_numeric = synthetic_numeric[common_cols].fillna(0)

    if len(real_numeric) > len(synthetic_numeric):
        real_numeric = real_numeric.sample(n=len(synthetic_numeric), random_state=42)

    real_labels = np.ones(len(real_numeric))
    synthetic_labels = np.zeros(len(synthetic_numeric))

    X = pd.concat([real_numeric, synthetic_numeric], ignore_index=True)
    y = np.concatenate([real_labels, synthetic_labels])

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
    detection_accuracy = scores.mean()

    realism_score = round((1 - (detection_accuracy - 0.5) * 2) * 100, 1)
    realism_score = max(0, min(100, realism_score))

    if realism_score >= 80:
        grade = "Excellent"
        color = "green"
    elif realism_score >= 60:
        grade = "Good"
        color = "yellow"
    else:
        grade = "Fair"
        color = "red"

    return {
        "realism_score": realism_score,
        "grade": grade,
        "color": color,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SynthIQ backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    summary = {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "preview": df.head(5).to_dict(orient="records")
    }

    return summary

@app.post("/analyse")
async def analyse_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    issues = detect_issues(df)

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "issues": issues,
        "issues_found": len(issues)
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

    if num_rows > 1000:
        num_rows = 1000

    # Parse and apply approved fixes
    try:
        fixes_list = [Fix(**f) for f in json.loads(fixes)]
        cleaned_df = apply_fixes(real_df, fixes_list)
    except Exception:
        cleaned_df = real_df

    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(cleaned_df)

        synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=False)
        synthesizer.fit(cleaned_df)

        synthetic_df = synthesizer.sample(num_rows=num_rows)
        score_data = calculate_realism_score(cleaned_df, synthetic_df)

        output = io.StringIO()
        synthetic_df.to_csv(output, index=False)
        csv_string = output.getvalue()

        return {
            "realism_score": score_data["realism_score"],
            "grade": score_data["grade"],
            "color": score_data["color"],
            "rows_generated": len(synthetic_df),
            "csv_data": csv_string
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))