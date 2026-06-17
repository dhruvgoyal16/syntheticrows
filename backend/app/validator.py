import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from .models import ValidationResult, ColumnQuality
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


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


def tstr_validation(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Train on Synthetic, Test on Real validation.
    Compares a model trained on synthetic data vs one trained on real data,
    both tested on held-out real data. Supports numeric AND categorical targets.
    """
    # Find target column if not provided
    if target_col is None:
        target_keywords = ["target", "label", "class", "outcome", "y", "output", "result"]
        for col in real_df.columns:
            if col.lower() in target_keywords:
                target_col = col
                break

    # Fall back to last binary column
    if target_col is None:
        for col in real_df.columns:
            if real_df[col].nunique() == 2:
                target_col = col
                break

    # Can't run TSTR without a target
    if target_col is None or target_col not in real_df.columns:
        return {"available": False, "reason": "No target column detected for TSTR validation"}

    try:
        # Features must be numeric; the TARGET may be numeric OR categorical
        # (we label-encode a categorical target so classification still works).
        real_numeric = real_df.select_dtypes(include=[np.number])

        # Feature columns = numeric columns excluding the target (if it's numeric)
        feature_cols = [c for c in real_numeric.columns if c != target_col]
        if len(feature_cols) == 0:
            return {"available": False, "reason": "No numeric feature columns found for ML-readiness testing"}

        # Synthetic must share those feature columns
        synth_numeric = synthetic_df.select_dtypes(include=[np.number])
        common_features = [c for c in feature_cols if c in synth_numeric.columns]
        if len(common_features) == 0:
            return {"available": False, "reason": "No matching numeric feature columns in synthetic data"}

        # The target must exist in both real and synthetic
        if target_col not in real_df.columns or target_col not in synthetic_df.columns:
            return {"available": False, "reason": "Target column missing from synthetic data"}

        # Encode the target consistently across real + synthetic so the same
        # class maps to the same number in both (handles string targets like
        # sentiment/category, and is harmless for already-numeric targets).
        le = LabelEncoder()
        combined_targets = pd.concat([
            real_df[target_col].astype(str),
            synthetic_df[target_col].astype(str),
        ], ignore_index=True)
        le.fit(combined_targets)

        X_real = real_df[common_features].fillna(0)
        y_real = le.transform(real_df[target_col].astype(str))

        X_synth = synth_numeric[common_features].fillna(0)
        y_synth = le.transform(synthetic_df[target_col].astype(str))

        # A target with only one class can't be used for classification
        if len(np.unique(y_real)) < 2:
            return {"available": False, "reason": "Target column has only one class — can't measure ML readiness"}

        # Split real data into train and test (stratify only if every class has >= 2 samples)
        _, counts = np.unique(y_real, return_counts=True)
        stratify = y_real if counts.min() >= 2 else None
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=stratify
        )

        # Train on REAL, test on REAL (baseline)
        clf_real = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_real.fit(X_real_train[common_features], y_real_train)
        real_real_acc = round(accuracy_score(y_real_test, clf_real.predict(X_real_test[common_features])) * 100, 1)

        # Train on SYNTHETIC, test on REAL (TSTR)
        clf_synth = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_synth.fit(X_synth, y_synth)
        synth_real_acc = round(accuracy_score(y_real_test, clf_synth.predict(X_real_test[common_features])) * 100, 1)

        # Calculate performance gap
        gap = round(real_real_acc - synth_real_acc, 1)
        gap_pct = round((abs(gap) / real_real_acc) * 100, 1) if real_real_acc > 0 else 0
        synthetic_better = synth_real_acc > real_real_acc

        # ── Honesty guard: is the Real→Real baseline actually meaningful? ──
        # If a classifier can reach ~real_real_acc just by always predicting the
        # majority class, a small synth-vs-real gap is NOT evidence the synthetic
        # data is good — both models are essentially guessing. In that case we
        # return an honest "inconclusive" verdict instead of celebrating.
        majority_class_rate = round(
            float(pd.Series(y_real_test).value_counts(normalize=True).iloc[0]) * 100, 1
        )
        baseline_is_weak = real_real_acc <= max(majority_class_rate + 5, 60)

        if baseline_is_weak:
            return {
                "available": True,
                "target_column": target_col,
                "real_real_accuracy": real_real_acc,
                "synth_real_accuracy": synth_real_acc,
                "performance_gap": gap,
                "performance_gap_pct": gap_pct,
                "grade": "Inconclusive",
                "color": "yellow",
                "interpretation": (
                    f"This comparison isn't very meaningful for this dataset. Even a model trained on "
                    f"real data only reaches {real_real_acc}% accuracy — about what you'd get by always "
                    f"guessing the most common class ({majority_class_rate}%). That usually means the "
                    f"target is hard to predict from these columns, or the classes are heavily imbalanced. "
                    f"Read the synthetic-vs-real numbers ({synth_real_acc}% vs {real_real_acc}%) with "
                    f"caution rather than as proof the synthetic data is good or bad."
                ),
            }

        # Grade the gap (only reached when the baseline is meaningful)
        if synthetic_better:
            tstr_grade = "Excellent"
            tstr_color = "green"
            interpretation = f"Exceptional result. A model trained on your synthetic data actually outperforms one trained on real data ({synth_real_acc}% vs {real_real_acc}%). Your synthetic data has effectively smoothed out noise in the original dataset, making it even more useful for ML training."
        elif gap_pct <= 5:
            tstr_grade = "Excellent"
            tstr_color = "green"
            interpretation = f"Outstanding result. A model trained on your synthetic data performs nearly identically to one trained on real data — only a {gap}% accuracy gap. Your synthetic data is production-ready and can fully replace real data for ML training."
        elif gap_pct <= 10:
            tstr_grade = "Excellent"
            tstr_color = "green"
            interpretation = f"Strong result. Your synthetic data produces a model that performs very close to one trained on real data — only a {gap}% accuracy gap. Your synthetic data is ready for ML training and most production use cases."
        elif gap_pct <= 20:
            tstr_grade = "Good"
            tstr_color = "yellow"
            interpretation = f"Good result. Your synthetic data produces a model with a {gap}% accuracy gap compared to real data. This is acceptable for training, augmentation, and testing. Consider approving more data quality fixes to further close the gap."
        elif gap_pct <= 35:
            tstr_grade = "Fair"
            tstr_color = "yellow"
            interpretation = f"Moderate result. There is a {gap}% accuracy gap between models trained on synthetic vs real data. Your synthetic data can be used for initial experimentation. Approve more quality fixes and regenerate to improve ML readiness."
        else:
            tstr_grade = "Poor"
            tstr_color = "red"
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


def validate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, target_col: str = None) -> ValidationResult:
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
    tstr = tstr_validation(real_df, synthetic_df, target_col)

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