import numpy as np
import pandas as pd

from logging_utils import logger

def remove_outliers_by_iqr(df, features_to_check, threshold=3):

    outlier_mask = pd.Series([False] * len(df))  # Initialize a mask of False values

    for col in features_to_check:
        # Step 1: Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        # Step 2: Calculate the IQR
        IQR = Q3 - Q1

        # Step 3: Calculate lower and upper bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Update the outlier mask for this column (True where outliers are found)
        outlier_mask |= ((df[col] < lower_bound) | (df[col] > upper_bound))

    # Remove outliers
    df = df[~outlier_mask]

    # Log details of dropped outliers
    outliers_count = outlier_mask.sum()
    outliers_pct = (outliers_count / df.index.size * 100)
    logger.debug("Dropping %s (%.1f%%) as outliers", outliers_count, outliers_pct)

    return df