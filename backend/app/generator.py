import pandas as pd
import numpy as np
import random
from .models import ModelType, DatasetProfile
from .router import select_model
from sdv.metadata import Metadata
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)
from .preprocessor import detect_skewed_columns, log_transform, reverse_log_transform, restore_dtypes, apply_domain_constraints, bootstrap_tiny_dataset

def encode_discrete_as_categorical(df: pd.DataFrame, profile: DatasetProfile) -> tuple[pd.DataFrame, list]:
    """
    Convert low-cardinality integer columns to string categoricals
    so SDV treats them properly during generation.
    """
    from .models import ColumnType
    df = df.copy()
    encoded_cols = []

    for col_profile in profile.columns:
        col = col_profile.name
        if col not in df.columns:
            continue
        if col_profile.col_type == ColumnType.NUMERICAL_DISCRETE and df[col].nunique() <= 10:
            df[col] = df[col].astype(str)
            encoded_cols.append(col)

    return df, encoded_cols


def decode_categorical_to_discrete(df: pd.DataFrame, encoded_cols: list, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string categoricals back to integers after generation.
    """
    df = df.copy()
    for col in encoded_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(original_df[col].mode()[0])
                df[col] = df[col].round().astype(int)
            except Exception:
                pass
    return df

def build_synthesizer(model_type: ModelType, metadata: Metadata, kwargs: dict):
    if model_type == ModelType.GAUSSIAN_COPULA:
        return GaussianCopulaSynthesizer(metadata)
    elif model_type == ModelType.CTGAN:
        epochs = kwargs.get("epochs", 300)
        return CTGANSynthesizer(metadata, epochs=epochs, verbose=False, cuda=False)
    elif model_type == ModelType.TVAE:
        return TVAESynthesizer(metadata)
    else:
        return CTGANSynthesizer(metadata, epochs=200, verbose=False, cuda=False)


def get_conditional_samples(
    df: pd.DataFrame,
    profile: DatasetProfile,
    synthesizer,
    num_rows: int
) -> pd.DataFrame:
    target_col = profile.target_column

    if target_col is None or target_col not in df.columns:
        return synthesizer.sample(num_rows=num_rows)

    class_counts = df[target_col].value_counts(normalize=True)
    frames = []

    for class_val, proportion in class_counts.items():
        n = max(1, round(num_rows * proportion))
        try:
            condition = pd.DataFrame({target_col: [class_val] * n})
            sample = synthesizer.sample_remaining_columns(condition)
            frames.append(sample)
        except Exception:
            sample = synthesizer.sample(num_rows=n)
            frames.append(sample)

    return pd.concat(frames, ignore_index=True)


def generate(
    df: pd.DataFrame,
    profile: DatasetProfile,
    num_rows: int,
    class_ratios: dict = {}
) -> tuple[pd.DataFrame, str]:
    """
    Returns (synthetic_df, model_name_used)
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    model_type, model_kwargs = select_model(profile)
    use_conditional = model_kwargs.pop("conditional", False)

    # For truly tiny datasets — use bootstrapping instead of generative models
    if len(df) < 50 and not class_ratios:
        synthetic_df = bootstrap_tiny_dataset(df, num_rows)
        synthetic_df = restore_dtypes(synthetic_df, df)
        synthetic_df = apply_domain_constraints(synthetic_df, df)
        return synthetic_df, "Bootstrap"
    
    # Encode discrete columns as categorical for better generation

    df_encoded, encoded_discrete_cols = encode_discrete_as_categorical(df, profile)

    # Detect and apply log transform for skewed columns
    skewed_cols = detect_skewed_columns(df_encoded)
    if skewed_cols:
        df_transformed = log_transform(df_encoded, skewed_cols)
    else:
        df_transformed = df_encoded.copy()

    metadata = Metadata.detect_from_dataframe(df_transformed)
    synthesizer = build_synthesizer(model_type, metadata, model_kwargs)
    synthesizer.fit(df_transformed)

    # Use custom class ratios if provided
    if class_ratios and profile.target_column and profile.target_column in df.columns:
        frames = []

        for class_val, count in class_ratios.items():
            try:
                col_dtype = df[profile.target_column].dtype
                if np.issubdtype(col_dtype, np.integer):
                    class_val_typed = int(class_val)
                elif np.issubdtype(col_dtype, np.floating):
                    class_val_typed = float(class_val)
                else:
                    class_val_typed = class_val

                n = int(count)

                class_df = df_transformed[
                    df_transformed[profile.target_column] == class_val_typed
                ]

                if len(class_df) < 10:
                    continue

                class_metadata = Metadata.detect_from_dataframe(class_df)
                class_synthesizer = build_synthesizer(model_type, class_metadata, model_kwargs.copy())
                class_synthesizer.fit(class_df)
                class_synthetic = class_synthesizer.sample(num_rows=n)
                class_synthetic[profile.target_column] = class_val_typed
                frames.append(class_synthetic)

            except Exception as e:
                print(f"Conditional generation failed for class {class_val}: {e}")
                continue

        if frames:
            synthetic_transformed = pd.concat(frames, ignore_index=True)
        else:
            synthetic_transformed = synthesizer.sample(num_rows=num_rows)

    # Use conditional sampling for imbalanced datasets
    elif use_conditional and profile.target_column:
        synthetic_transformed = get_conditional_samples(
            df_transformed, profile, synthesizer, num_rows
        )
    else:
        synthetic_transformed = synthesizer.sample(num_rows=num_rows)

    # Clip synthetic log values to real log range before reversing
    if skewed_cols:
        for col in skewed_cols:
            if col in synthetic_transformed.columns and col in df_transformed.columns:
                real_log_min = df_transformed[col].min()
                real_log_max = df_transformed[col].max()
                synthetic_transformed[col] = synthetic_transformed[col].clip(
                    lower=real_log_min,
                    upper=real_log_max
                )
        synthetic_df = reverse_log_transform(synthetic_transformed, skewed_cols)
    else:
        synthetic_df = synthetic_transformed

    # Decode categorical back to discrete integers
    if encoded_discrete_cols:
        synthetic_df = decode_categorical_to_discrete(synthetic_df, encoded_discrete_cols, df)

    # Restore integer types and clip to original ranges
    synthetic_df = restore_dtypes(synthetic_df, df)

    # Apply domain constraints
    synthetic_df = apply_domain_constraints(synthetic_df, df)

    return synthetic_df, model_type.value