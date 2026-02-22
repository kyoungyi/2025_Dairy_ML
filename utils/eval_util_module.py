
import sys
import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

def percentile_based_pdp_multi(model, X, features, n_bins=20, sample_size=100_000):
    # Sample the data to speed things up
    X_sample = X.sample(sample_size, random_state=42).copy()
    results = []
    
    for feature in features:
        # Compute bin edges for percentiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(X_sample[feature], percentiles)
        
        # Assign bins
        X_sample['bin'] = pd.cut(X_sample[feature], bins=bins, include_lowest=True, duplicates='drop')

        # Group and compute PDP values
        grouped = (
            X_sample
            .groupby('bin')
            .apply(lambda df: pd.Series({
                'feature': feature,
                'mean_feature_value': df[feature].mean(),
                'mean_prediction': model.predict(xgb.DMatrix(df.drop(columns='bin'))).mean()
            }))
        ).reset_index(drop=True)

        results.append(grouped)

    return pd.concat(results, ignore_index=True)

#def weighted_mse(y_true, y_pred):
#    """
#    Compute the weighted Normalized Mean Squared Error (NMSE).
#
#    Parameters:
#    - y_true: array-like, true target values
#    - y_pred: array-like, predicted values
#    - sample_weight: array-like, sample weights
#
#    Returns:
#    - weighted NMSE (float)
#    """
#    # Weighted mean of y_true
#    y_mean = np.average(y_true, weights=sample_weight)
#
#    # Weighted MSE (numerator)
#    mse = np.average((y_true - y_pred)**2, weights=sample_weight)
#
#    # Weighted variance (denominator)
#    var = np.average((y_true - y_mean)**2, weights=sample_weight)
#
#    # Return NMSE (no epsilon needed if var > 0 is guaranteed)
#    return mse / var

#def conditional_mse(y_true, y_pred, q=0.2):
#    low_thresh = np.quantile(y_true, q)
#    high_thresh = np.quantile(y_true, 1 - q)
#    low_mask = y_true < low_thresh
#    high_mask = y_true > high_thresh
#    low_mse = np.mean((y_true[low_mask] - y_pred[low_mask])**2)
#    high_mse = np.mean((y_true[high_mask] - y_pred[high_mask])**2)
#    return low_mse, high_mse

def conditional_mse(y_true, y_pred, q=0.2, extreme_clip=0.01):
    low_q = q
    high_q = 1-low_q
    # Find extreme thresholds
    low_extreme_thresh = np.quantile(y_true, extreme_clip)
    high_extreme_thresh = np.quantile(y_true, 1 - extreme_clip)
    
    # Find tail thresholds
    low_tail_thresh = np.quantile(y_true, low_q)
    high_tail_thresh = np.quantile(y_true, high_q)

    # Masks
    low_tail_mask = (y_true >= low_extreme_thresh) & (y_true <= low_tail_thresh)
    high_tail_mask = (y_true <= high_extreme_thresh) & (y_true >= high_tail_thresh)

    # Calculate MSE separately for low and high (excluding 0–1% and 99–100%)
    low_tail_mse = np.mean((y_true[low_tail_mask] - y_pred[low_tail_mask])**2)
    high_tail_mse = np.mean((y_true[high_tail_mask] - y_pred[high_tail_mask])**2)

    return low_tail_mse, high_tail_mse


def evaluate(target, fitted_value):
    ### Cow-level:
    cow_mse = mean_squared_error(target, fitted_value)
    cow_mae = mean_absolute_error(target, fitted_value)
    cow_condition = conditional_mse(target, fitted_value)
    cow_pear_r = stats.pearsonr(target, fitted_value)[0]
    cow_r2 = r2_score(target, fitted_value)
    print('########## Model Performance#########')
    print(f'MSE: {cow_mse:.5f}')
    print(f'MAE: {cow_mae:.5f}')
    print(f"Low_20% MSE: {cow_condition[0]:.5f}")
    print(f"High_20% MSE: {cow_condition[1]:.5f}")
    print(f"r2: {cow_r2:.5f}")
    print(f'pearson_r : {cow_pear_r:.5f}')
    return {
        'mse': cow_mse,
        'mae': cow_mae,
        'low20%_mse': cow_condition[0],
        'high20%_mse': cow_condition[1],
        'r2':cow_r2,
        'pearson_r': cow_pear_r
    }


