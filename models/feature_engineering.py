import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class PASFeatureEngineer:
    """
    Feature Engineering for PAS Forecasting (small datasets)
    Creates stable temporal and statistical features.
    """

    def __init__(self):
        pass

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate safe and consistent features for small datasets"""
        logger.info("🔧 Starting feature engineering...")
        df = df.copy()

        # --- Basic validation ---
        if "timestamp" not in df.columns or "entity_id" not in df.columns:
            raise ValueError("Missing required columns: 'timestamp' or 'entity_id'")

        # --- Date-related features ---
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
        df["time_idx"] = df.groupby("entity_id").cumcount().astype(int)

        df["year"] = df["timestamp"].dt.year
        df["month_num"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter
        df["days_in_month"] = df["timestamp"].dt.days_in_month
        df["is_year_end"] = (df["month_num"] == 12).astype(int)
        df["is_year_start"] = (df["month_num"] == 1).astype(int)

        # --- Cyclic encodings for time ---
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
        df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

        # --- Rolling statistics (safe for small data) ---
        if "spend_amount" in df.columns:
            df["spend_amount_mean"] = df.groupby("entity_id")["spend_amount"].transform("mean")
            df["spend_amount_std"] = df.groupby("entity_id")["spend_amount"].transform("std").fillna(0)
            df["spend_amount_min"] = df.groupby("entity_id")["spend_amount"].transform("min")
            df["spend_amount_max"] = df.groupby("entity_id")["spend_amount"].transform("max")
            df["spend_amount_ema_0_3"] = df.groupby("entity_id")["spend_amount"].transform(
                lambda x: x.ewm(span=max(2, len(x) // 2), adjust=False).mean()
            )
        else:
            logger.warning("⚠️ Missing 'spend_amount' column — skipping spend-related features.")

        # --- Transaction-level derived features ---
        if "transaction_count" in df.columns:
            df["transaction_count_mean"] = df.groupby("entity_id")["transaction_count"].transform("mean")
            df["spend_per_transaction"] = np.where(
                df["transaction_count"] > 0,
                df["spend_amount"] / df["transaction_count"],
                df["spend_amount"],
            )
        else:
            df["transaction_count"] = 0
            df["spend_per_transaction"] = df["spend_amount"]

        # --- Final cleanup ---
        df = df.fillna(0)
        for col in ["month", "quarter", "year"]:
         if col in df.columns:
           df[col] = df[col].astype(str)

        return df
        # categorical_cols = ["month", "quarter", "year"]
        # for col in categorical_cols:
        #   if col in df.columns:
        #      df[col] = df[col].astype(str)

        # logger.info(f"✅ Feature engineering completed. Final shape: {df.shape}")
        # return df

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return summary of generated features"""
        return pd.DataFrame({
            "feature": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing": [df[c].isna().sum() for c in df.columns],
        })
