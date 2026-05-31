import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from .models import ValidationResult, ColumnQuality


def distinguishability_score(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    real_num = real_df.select_dtypes(include=[np.number])
    synth_num = synthetic_df.select_dtypes(include=[np.number])

    common_cols = list(set(real_num.columns) & set(synth_num.columns))
    if not common_cols:
        return 50.0

    real_num = real_num[common_cols].fillna(0)
    synth_num = synth_num[common_cols].fillna(0)

    if len(real_num) > len(synth_num):
        real_num = real_num.sample(n=len(synth_num), random_state=42)

    X = pd.concat([real_num, synth_num], ignore_index=True)
    y = [1] * len(real_num) + [0] * len(synth_num)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(clf, X_scaled, y, cv=3, scoring="accuracy")
    detection_accuracy = scores.mean()

    score = (1 - (detection_accuracy - 0.5) * 2) * 100
    return round(max(0, min(100, score)), 1)


def statistical_similarity_score(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    real_num = real_df.select_dtypes(include=[np.number])
    synth_num = synthetic_df.select_dtypes(include=[np.number])

    common_cols = list(set(real_num.columns) & set(synth_num.columns))
    if not common_cols:
        return 50.0

    scores = []
    for col in common_cols:
        real_mean = real_num[col].mean()
        synth_mean = synth_num[col].mean()
        real_std = real_num[col].std()
        synth_std = synth_num[col].std()

        # Mean similarity
        if real_mean != 0:
            mean_diff = abs(real_mean - synth_mean) / abs(real_mean)
            mean_score = max(0, 1 - mean_diff)
        else:
            mean_score = 1.0 if synth_mean == 0 else 0.5

        # Std similarity
        if real_std != 0:
            std_diff = abs(real_std - synth_std) / abs(real_std)
            std_score = max(0, 1 - std_diff)
        else:
            std_score = 1.0 if synth_std == 0 else 0.5

        scores.append((mean_score + std_score) / 2)

    return round(np.mean(scores) * 100, 1)


def coverage_score(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    real_num = real_df.select_dtypes(include=[np.number])
    synth_num = synthetic_df.select_dtypes(include=[np.number])

    common_cols = list(set(real_num.columns) & set(synth_num.columns))
    if not common_cols:
        return 50.0

    scores = []
    for col in common_cols:
        real_min, real_max = real_num[col].min(), real_num[col].max()
        synth_min, synth_max = synth_num[col].min(), synth_num[col].max()

        real_range = real_max - real_min
        if real_range == 0:
            scores.append(1.0)
            continue

        # How much of the real range is covered by synthetic
        overlap_min = max(real_min, synth_min)
        overlap_max = min(real_max, synth_max)
        overlap = max(0, overlap_max - overlap_min)
        coverage = overlap / real_range
        scores.append(coverage)

    return round(np.mean(scores) * 100, 1)


def column_quality_report(real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    real_num = real_df.select_dtypes(include=[np.number])
    synth_num = synthetic_df.select_dtypes(include=[np.number])
    common_cols = list(set(real_num.columns) & set(synth_num.columns))

    report = []
    for col in common_cols:
        real_mean = real_num[col].mean()
        synth_mean = synth_num[col].mean()
        real_std = real_num[col].std()
        synth_std = synth_num[col].std()

        mean_diff_pct = round(abs(real_mean - synth_mean) / abs(real_mean) * 100, 1) if real_mean != 0 else 0.0
        std_diff_pct = round(abs(real_std - synth_std) / abs(real_std) * 100, 1) if real_std != 0 else 0.0

        col_score = max(0, 100 - (mean_diff_pct + std_diff_pct) / 2)

        if col_score >= 80:
            grade = "Excellent"
        elif col_score >= 60:
            grade = "Good"
        elif col_score >= 40:
            grade = "Fair"
        else:
            grade = "Poor"

        report.append(ColumnQuality(
            column=col,
            score=round(col_score, 1),
            grade=grade,
            mean_diff_pct=mean_diff_pct,
            std_diff_pct=std_diff_pct
        ))

    return sorted(report, key=lambda x: x.score, reverse=True)


def validate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> ValidationResult:
    d_score = distinguishability_score(real_df, synthetic_df)
    s_score = statistical_similarity_score(real_df, synthetic_df)
    c_score = coverage_score(real_df, synthetic_df)

    # Weighted final score
    final = round(d_score * 0.2 + s_score * 0.5 + c_score * 0.3, 1)
    if final >= 80:
        grade = "Excellent"
        color = "green"
    elif final >= 60:
        grade = "Good"
        color = "yellow"
    else:
        grade = "Fair"
        color = "red"

    col_quality = column_quality_report(real_df, synthetic_df)

    return ValidationResult(
        distinguishability_score=d_score,
        statistical_score=s_score,
        coverage_score=c_score,
        final_score=final,
        grade=grade,
        color=color,
        column_quality=col_quality
    )