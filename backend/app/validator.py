import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from .models import ValidationResult, ColumnQuality
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score


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

        if real_mean != 0:
            mean_diff = abs(real_mean - synth_mean) / abs(real_mean)
            mean_score = max(0, 1 - mean_diff)
        else:
            mean_score = 1.0 if synth_mean == 0 else 0.5

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



def _prepare_features(real_df, synthetic_df, target_col, max_card=20):
    """Build an aligned feature matrix for real & synthetic, one-hot encoding
    low-cardinality categorical FEATURE columns (excluding the target) so the
    models can use predictive categoricals like smoker/region — not just numeric
    columns. High-cardinality categoricals (> max_card uniques) are skipped to
    avoid feature-space blowup. Returns (X_real, X_synth) with identical columns."""
    feat_cols = [c for c in real_df.columns if c != target_col]
    cat_cols = []
    usable = []
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(real_df[c]):
            usable.append(c)
        elif c in synthetic_df.columns and real_df[c].nunique(dropna=True) <= max_card:
            usable.append(c)
            cat_cols.append(c)
    usable = [c for c in usable if c in synthetic_df.columns]
    cat_cols = [c for c in cat_cols if c in usable]
    if not usable:
        return None, None
    R = real_df[usable].copy(); R["__src__"] = "r"
    S = synthetic_df[usable].copy(); S["__src__"] = "s"
    both = pd.concat([R, S], ignore_index=True)
    if cat_cols:
        both = pd.get_dummies(both, columns=cat_cols, drop_first=True)
    X_real = both[both["__src__"] == "r"].drop(columns=["__src__"]).fillna(0)
    X_synth = both[both["__src__"] == "s"].drop(columns=["__src__"]).fillna(0)
    return X_real, X_synth


def _tstr_regression(real_df, synthetic_df, target_col, common_features):
    """Regression TSTR: train regressor on real vs synthetic, test on held-out
    real data, compare R^2 (1.0 perfect, 0 = no better than predicting the mean)."""
    X_real, X_synth = _prepare_features(real_df, synthetic_df, target_col)
    if X_real is None or X_real.shape[1] == 0:
        return {"available": False, "reason": "No usable feature columns for regression ML-readiness"}
    y_real = pd.to_numeric(real_df[target_col], errors="coerce")
    y_synth = pd.to_numeric(synthetic_df[target_col], errors="coerce")

    real_mask = y_real.notna()
    synth_mask = y_synth.notna()
    X_real, y_real = X_real[real_mask.values], y_real[real_mask]
    X_synth, y_synth = X_synth[synth_mask.values], y_synth[synth_mask]

    if len(X_real) < 10 or len(X_synth) < 10:
        return {"available": False, "reason": "Not enough rows to measure regression ML-readiness"}

    Xtr, Xte, ytr, yte = train_test_split(X_real, y_real, test_size=0.3, random_state=42)

    m_real = RandomForestRegressor(n_estimators=50, random_state=42)
    m_real.fit(Xtr, ytr)
    r2_real = r2_score(yte, m_real.predict(Xte))

    m_synth = RandomForestRegressor(n_estimators=50, random_state=42)
    m_synth.fit(X_synth, y_synth)
    r2_synth = r2_score(yte, m_synth.predict(Xte))

    real_score = round(max(0.0, min(1.0, r2_real)) * 100, 1)
    synth_score = round(max(0.0, min(1.0, r2_synth)) * 100, 1)
    gap = round(real_score - synth_score, 1)
    synthetic_better = synth_score > real_score

    if r2_real < 0.1:
        return {
            "available": True,
            "target_column": target_col,
            "is_regression": True,
            "real_real_accuracy": real_score,
            "synth_real_accuracy": synth_score,
            "performance_gap": gap,
            "performance_gap_pct": 0,
            "grade": "Inconclusive",
            "color": "neutral",
            "interpretation": (
                f"This result is inconclusive — and that's informational, not a problem with your data. "
                f"Even a model trained on real data barely predicts \"{target_col}\" (an R\u00b2 fit score of "
                f"only {real_score}/100). That usually means this column is hard to predict from the other "
                f"columns, or the relationship is weak. With such a low baseline, comparing synthetic to real "
                f"doesn't tell us much. Try a target that's more related to your other columns."
            ),
        }

    gap_pct = round((abs(gap) / real_score) * 100, 1) if real_score > 0 else 0

    if synthetic_better:
        grade, color = "Excellent", "green"
        interp = (f"Exceptional result. A model trained on your synthetic data predicts \"{target_col}\" "
                  f"slightly better than one trained on real data (fit score {synth_score} vs {real_score} "
                  f"out of 100). Your synthetic data preserved the relationships well — it's ready for ML training.")
    elif gap_pct <= 5:
        grade, color = "Excellent", "green"
        interp = (f"Outstanding result. A model trained on your synthetic data predicts \"{target_col}\" almost "
                  f"as well as one trained on real data (fit score {synth_score} vs {real_score} out of 100). "
                  f"Your synthetic data is production-ready for ML training.")
    elif gap_pct <= 10:
        grade, color = "Excellent", "green"
        interp = (f"Strong result. Your synthetic data predicts \"{target_col}\" very close to real data "
                  f"(fit score {synth_score} vs {real_score} out of 100). It's ready for ML training and most "
                  f"production use cases.")
    elif gap_pct <= 20:
        grade, color = "Good", "yellow"
        interp = (f"Good result. A model trained on your synthetic data predicts \"{target_col}\" with a fit "
                  f"score of {synth_score} vs {real_score} on real data. This is acceptable for training, "
                  f"augmentation, and testing. Approving more data-quality fixes can close the gap.")
    elif gap_pct <= 35:
        grade, color = "Fair", "yellow"
        interp = (f"Moderate result. There's a meaningful gap in how well synthetic-trained vs real-trained "
                  f"models predict \"{target_col}\" (fit score {synth_score} vs {real_score}). Usable for early "
                  f"experimentation; approve more fixes and regenerate to improve.")
    else:
        grade, color = "Poor", "red"
        interp = (f"The gap between synthetic-trained and real-trained models predicting \"{target_col}\" "
                  f"(fit score {synth_score} vs {real_score}) suggests the synthetic data needs improvement "
                  f"before ML training. Review and approve the recommended fixes, then regenerate.")

    return {
        "available": True,
        "target_column": target_col,
        "is_regression": True,
        "real_real_accuracy": real_score,
        "synth_real_accuracy": synth_score,
        "performance_gap": gap,
        "performance_gap_pct": gap_pct,
        "grade": grade,
        "color": color,
        "interpretation": interp,
    }


def tstr_validation(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                    target_col: str = None, target_type: str = None) -> dict:
    """
    Train on Synthetic, Test on Real validation.
    Routes to a regression path (R^2) when target_type == 'regression',
    otherwise runs classification (accuracy), as before.
    """
    if target_col is None:
        target_keywords = ["target", "label", "class", "outcome", "y", "output", "result"]
        for col in real_df.columns:
            if col.lower() in target_keywords:
                target_col = col
                break

    if target_col is None:
        for col in real_df.columns:
            if real_df[col].nunique() == 2:
                target_col = col
                break

    if target_col is None or target_col not in real_df.columns:
        return {"available": False, "reason": "No target column detected for TSTR validation"}

    try:
        real_numeric = real_df.select_dtypes(include=[np.number])
        feature_cols = [c for c in real_numeric.columns if c != target_col]
        if len(feature_cols) == 0:
            return {"available": False, "reason": "No numeric feature columns found for ML-readiness testing"}

        synth_numeric = synthetic_df.select_dtypes(include=[np.number])
        common_features = [c for c in feature_cols if c in synth_numeric.columns]
        if len(common_features) == 0:
            return {"available": False, "reason": "No matching numeric feature columns in synthetic data"}

        if target_col not in real_df.columns or target_col not in synthetic_df.columns:
            return {"available": False, "reason": "Target column missing from synthetic data"}

        # ── Regression path ──
        if target_type == "regression":
            return _tstr_regression(real_df, synthetic_df, target_col, common_features)

        # ── Classification path ──
        le = LabelEncoder()
        combined_targets = pd.concat([
            real_df[target_col].astype(str),
            synthetic_df[target_col].astype(str),
        ], ignore_index=True)
        le.fit(combined_targets)

        # Use one-hot-aligned features (numeric + low-card categoricals), so
        # predictive categorical columns are not silently dropped.
        X_real, X_synth = _prepare_features(real_df, synthetic_df, target_col)
        if X_real is None or X_real.shape[1] == 0:
            return {"available": False, "reason": "No usable feature columns for ML-readiness testing"}
        y_real = le.transform(real_df[target_col].astype(str))
        y_synth = le.transform(synthetic_df[target_col].astype(str))

        if len(np.unique(y_real)) < 2:
            return {"available": False, "reason": "Target column has only one class — can't measure ML readiness"}

        _, counts = np.unique(y_real, return_counts=True)
        stratify = y_real if counts.min() >= 2 else None
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=stratify
        )

        clf_real = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_real.fit(X_real_train, y_real_train)
        real_real_acc = round(accuracy_score(y_real_test, clf_real.predict(X_real_test)) * 100, 1)

        clf_synth = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_synth.fit(X_synth, y_synth)
        synth_real_acc = round(accuracy_score(y_real_test, clf_synth.predict(X_real_test)) * 100, 1)

        gap = round(real_real_acc - synth_real_acc, 1)
        gap_pct = round((abs(gap) / real_real_acc) * 100, 1) if real_real_acc > 0 else 0
        synthetic_better = synth_real_acc > real_real_acc

        majority_class_rate = round(
            float(pd.Series(y_real_test).value_counts(normalize=True).iloc[0]) * 100, 1
        )
        baseline_is_weak = real_real_acc <= majority_class_rate + 2

        if baseline_is_weak:
            return {
                "available": True,
                "target_column": target_col,
                "real_real_accuracy": real_real_acc,
                "synth_real_accuracy": synth_real_acc,
                "performance_gap": gap,
                "performance_gap_pct": gap_pct,
                "grade": "Inconclusive",
                "color": "neutral",
                "interpretation": (
                    f"This result is inconclusive — and that's informational, not a problem with your data. "
                    f"Even a model trained on real data only reaches {real_real_acc}%, which is about what "
                    f"you'd get by always guessing the most common class ({majority_class_rate}%). So neither "
                    f"model really had to learn — which means even a small accuracy gap here ({gap}%) doesn't "
                    f"tell us much, because both models are mostly guessing the majority class. This usually "
                    f"happens when the target is heavily imbalanced or hard to predict from the other columns. "
                    f"To get a meaningful ML-readiness read, try a more balanced target or one more related to "
                    f"your features."
                ),
            }

        if synthetic_better:
            tstr_grade = "Excellent"; tstr_color = "green"
            interpretation = f"Exceptional result. A model trained on your synthetic data actually outperforms one trained on real data ({synth_real_acc}% vs {real_real_acc}%). Your synthetic data has effectively smoothed out noise in the original dataset, making it even more useful for ML training."
        elif gap_pct <= 5:
            tstr_grade = "Excellent"; tstr_color = "green"
            interpretation = f"Outstanding result. A model trained on your synthetic data performs nearly identically to one trained on real data — only a {gap}% accuracy gap. Your synthetic data is production-ready and can fully replace real data for ML training."
        elif gap_pct <= 10:
            tstr_grade = "Excellent"; tstr_color = "green"
            interpretation = f"Strong result. Your synthetic data produces a model that performs very close to one trained on real data — only a {gap}% accuracy gap. Your synthetic data is ready for ML training and most production use cases."
        elif gap_pct <= 20:
            tstr_grade = "Good"; tstr_color = "yellow"
            interpretation = f"Good result. Your synthetic data produces a model with a {gap}% accuracy gap compared to real data. This is acceptable for training, augmentation, and testing. Consider approving more data quality fixes to further close the gap."
        elif gap_pct <= 35:
            tstr_grade = "Fair"; tstr_color = "yellow"
            interpretation = f"Moderate result. There is a {gap}% accuracy gap between models trained on synthetic vs real data. Your synthetic data can be used for initial experimentation. Approve more quality fixes and regenerate to improve ML readiness."
        else:
            tstr_grade = "Poor"; tstr_color = "red"
            interpretation = f"The {gap}% accuracy gap suggests your synthetic data needs improvement before use in ML training. Review and approve all recommended data quality fixes, then regenerate for better results."

        return {
            "available": True,
            "target_column": target_col,
            "real_real_accuracy": real_real_acc,
            "synth_real_accuracy": synth_real_acc,
            "performance_gap": gap,
            "performance_gap_pct": gap_pct,
            "grade": tstr_grade,
            "color": tstr_color,
            "interpretation": interpretation
        }

    except Exception as e:
        return {"available": False, "reason": f"TSTR validation failed: {str(e)}"}


def validate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
             target_col: str = None, target_type: str = None) -> ValidationResult:
    d_score = distinguishability_score(real_df, synthetic_df)
    s_score = statistical_similarity_score(real_df, synthetic_df)
    c_score = coverage_score(real_df, synthetic_df)

    final = round(d_score * 0.2 + s_score * 0.5 + c_score * 0.3, 1)

    if final >= 80:
        grade = "Excellent"; color = "green"
    elif final >= 60:
        grade = "Good"; color = "yellow"
    else:
        grade = "Fair"; color = "red"

    col_quality = column_quality_report(real_df, synthetic_df)
    tstr = tstr_validation(real_df, synthetic_df, target_col, target_type)

    return ValidationResult(
        distinguishability_score=d_score,
        statistical_score=s_score,
        coverage_score=c_score,
        final_score=final,
        grade=grade,
        color=color,
        column_quality=col_quality,
        tstr=tstr
    )