import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy import stats
import seaborn as sns
import glob
import os, sys, gc
from pathlib import Path
import ast
from scipy.signal import find_peaks
from typing import Dict, List, Tuple

param = {'random':[40,41,42,43],
         'control':['lon','lat','month_cos','month_sin','lac_dim']}

feat_var = ['tmin','tmax_ssrd','rh_am','ag_wind_2m']
target = 'herd_milk_resid'
warm_state = ['CA','AZ','NM','TX','KS','OK','MO','AR','LA','KY','TN','MS','AL','GA','FL','SC','NC','VA']
control_var = param['control']
sub_ori_cols = control_var + feat_var
sub_cols = ['night_' + item for item in sub_ori_cols]

PAD = 1e-6

quantiles = np.concatenate([[0.0,0.01,0.03],np.arange(0.05,0.95,0.03),[0.95,0.97,0.99,1.0]])
mid_quant = (quantiles[:-1] + quantiles[1:]) / 2  # Midpoints of quantile bins

# Variables that don't use quantile-based binning
non_quantile_vars = ['month_cos', 'month_sin', 'lac_dim', 'lat', 'lon']

def compute_ale(data, var, bin_edges, bin_vals, model, feature_cols, weight_col, region):
    """Compute ALE curve for a variable across bins."""
    ale_vals, weight_sum, sample_num, left_full, right_full = [], [], [], [], []
    
    for i in range(len(bin_edges) - 1):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = data.loc[(data[var] > left) & (data[var] <= right)].copy()
        
        if mask.shape[0] == 0:
            ale_vals.append(0.0)
            weight_sum.append(0.0)
            sample_num.append(0)
            left_full.append(left)
            right_full.append(right)
            continue
        
        weights = mask[weight_col].values
        weight_sum.append(weights.sum())
        sample_num.append(mask.shape[0])
        left_full.append(left)
        right_full.append(right)
        
        # Predict at boundaries
        for value in [left, right]:
            tmp = mask.copy()
            tmp[var] = value
            mask[value] = model.predict(xgb.DMatrix(tmp[feature_cols], nthread=-1), pred_contribs=False)
            del tmp
            gc.collect()
        
        ale_vals.append(np.average(mask[right] - mask[left], weights=weights))
        del mask
        gc.collect()

    ale_curve = np.cumsum(ale_vals)
    ale_curve -= np.average(ale_curve, weights=weight_sum)
    
    # Prepare result dictionary
    result_dict = {
        'values': bin_vals,
        'ale': ale_curve,
        'cow_weight': weight_sum,
        'n': sample_num,
        'feat_abv': var,
        'div': region,
        'left': left_full,
        'right': right_full
    }
    
    # Add quantile column only for quantile-based variables
    if var not in non_quantile_vars:
        # Ensure length matches
        n_bins = len(ale_curve)
        if n_bins <= len(mid_quant):
            result_dict['quantile'] = mid_quant[:n_bins]
        else:
            # If somehow more bins than quantiles, pad with NaN
            result_dict['quantile'] = list(mid_quant) + [np.nan] * (n_bins - len(mid_quant))
    else:
        # For non-quantile variables, use NaN or omit
        result_dict['quantile'] = [np.nan] * len(ale_curve)
    
    return pd.DataFrame(result_dict)



# Step 4: Check if each peak is isolated and surrounded by negatives
def is_isolated_or_noisy_peak(ale_values, peak_idx, window=2, 
                               negative_threshold=0.5,
                               neighbor_diff_threshold=1.6):  # Changed parameter
    """
    Exclude peak if:
    1. Surrounded by negative bins
    2. Large difference from surrounding bins (noisy spike)
    """
    n = len(ale_values)
    peak_ale = ale_values[peak_idx]
    
    # Get surrounding bins
    left_start = max(0, peak_idx - window)
    right_end = min(n, peak_idx + window + 1)
    
    surrounding_bins = np.concatenate([
        ale_values[left_start:peak_idx],
        ale_values[peak_idx+1:right_end]
    ])
    
    if len(surrounding_bins) == 0:
        return False
    
    # Check 1: Are surrounding bins mostly negative?
    negative_ratio = np.sum(surrounding_bins < 0) / len(surrounding_bins)
    if negative_ratio > negative_threshold:
        print(f"EXCLUDED: {negative_ratio:.1f}% surrounding bins are negative")
        return True
    
    # Check 2: Look at ONLY immediate neighbors (peak ± 1) for noise
    immediate_neighbors = []
    
    # Left neighbor (peak - 1)
    if peak_idx > 0:
        immediate_neighbors.append(ale_values[peak_idx - 1])
    
    # Right neighbor (peak + 1)
    if peak_idx < n - 1:
        immediate_neighbors.append(ale_values[peak_idx + 1])
    
    if len(immediate_neighbors) > 0:
        # Calculate how much neighbors differ from peak
        diffs_from_peak = np.abs(np.array(immediate_neighbors) - peak_ale)
        max_diff = np.max(diffs_from_peak)
        print(max_diff, peak_ale)
        relative_diff = max_diff / abs(peak_ale) if peak_ale != 0 else np.inf
        
        if relative_diff > neighbor_diff_threshold:
            print(f"EXCLUDED: Immediate neighbors differ {relative_diff:.1f}% from peak")
            print(f"Peak ALE = {peak_ale:.6f}")
            print(f"Neighbors = {[f'{x:.3f}' for x in immediate_neighbors]}")
            return True
    
    return False  # Keep: stable positive region

# Step 6: Build range starting from highest peak, expanding as needed
def expand_from_peak(ale_values, peak_idx, n_bins_expand):
    """
    Expand from a peak by n_bins_expand in each direction.
    Stop at negative values or boundaries.
    
    Returns: (left_idx, right_idx)
    """
    n = len(ale_values)
    left_idx = peak_idx
    right_idx = peak_idx
    
    # Expand left
    for i in range(n_bins_expand):
        if left_idx > 0 and ale_values[left_idx - 1] > 0:
            left_idx -= 1
        else:
            break  # Stop at negative or boundary
    
    # Expand right
    for i in range(n_bins_expand):
        if right_idx < n - 1 and ale_values[right_idx + 1] > 0:
            right_idx += 1
        else:
            break  # Stop at negative or boundary
    
    return left_idx, right_idx

def expand_toward_peak(ale_values, left_idx, right_idx, target_peak_idx, n_bins_expand):
    """
    Expand the range toward a target peak by n_bins_expand.
    Stop at negative values or boundaries.
    """
    n = len(ale_values)
    
    if target_peak_idx < left_idx:
        # Expand left toward target
        for i in range(n_bins_expand):
            if left_idx > 0 and ale_values[left_idx - 1] > 0:
                left_idx -= 1
                if left_idx <= target_peak_idx:  # Reached the target
                    break
            else:
                break  # Stop at negative or boundary
    
    elif target_peak_idx > right_idx:
        # Expand right toward target
        for i in range(n_bins_expand):
            if right_idx < n - 1 and ale_values[right_idx + 1] > 0:
                right_idx += 1
                if right_idx >= target_peak_idx:  # Reached the target
                    break
            else:
                break  # Stop at negative or boundary
    
    return left_idx, right_idx

def build_optimal_ranges(ale_values, filtered_data, peak_info_list, similarity_threshold=0.8):
    """
    Build optimal ranges with specific hierarchical logic:
    
    Range 1 (from peak 1): Can include peaks 1, 2, 3 (max)
    Range 2 (from peak 2 if not in range 1): Can include peaks 2, 3 (max)
    Range 3 (from peak 3 if not in range 2 and similar enough): Only peak 3
    """
    ranges = []
    included_peaks = set()
    
    # Range 1: Start from highest peak (peak 1)
    if len(peak_info_list) >= 1:
        peak1 = peak_info_list[0]
        print(f"\n{'='*60}")
        print(f"RANGE 1: Starting from Peak 1 (index {peak1['index']}, ALE={peak1['ale']:.6f})")
        print('='*60)
        
        # Initial expansion ±2 bins
        left_idx, right_idx = expand_from_peak(ale_values, peak1['index'], n_bins_expand=2)
        peaks_in_range = [peak1['index']]
        included_peaks.add(peak1['index'])
        
        print(f"Initial expansion (±2): indices [{left_idx}:{right_idx+1}]")
        print(f"  Values: [{filtered_data.iloc[left_idx]['values']:.3f}, "
              f"{filtered_data.iloc[right_idx]['values']:.3f}]")
        
        # Check if peak 2 is included
        if len(peak_info_list) >= 2:
            peak2 = peak_info_list[1]
            peak2_idx = peak2['index']
            
            if left_idx <= peak2_idx <= right_idx:
                print(f"\nPeak 2 (index {peak2_idx}, ALE={peak2['ale']:.6f}) IS IN range")
                print(f"→ Expanding ±3 bins toward Peak 2")
                
                left_idx, right_idx = expand_toward_peak(ale_values, left_idx, right_idx, 
                                                         peak2_idx, n_bins_expand=3)
                peaks_in_range.append(peak2_idx)
                included_peaks.add(peak2_idx)
                
                print(f"  After expansion: indices [{left_idx}:{right_idx+1}]")
                print(f"  Values: [{filtered_data.iloc[left_idx]['values']:.3f}, "
                      f"{filtered_data.iloc[right_idx]['values']:.3f}]")
                
                # Check if peak 3 is now included
                if len(peak_info_list) >= 3:
                    peak3 = peak_info_list[2]
                    peak3_idx = peak3['index']
                    
                    if left_idx <= peak3_idx <= right_idx:
                        print(f"\nPeak 3 (index {peak3_idx}, ALE={peak3['ale']:.6f}) IS IN range")
                        print(f"→ Expanding ±3 bins toward Peak 3")
                        
                        left_idx, right_idx = expand_toward_peak(ale_values, left_idx, right_idx,
                                                                 peak3_idx, n_bins_expand=3)
                        peaks_in_range.append(peak3_idx)
                        included_peaks.add(peak3_idx)
                        
                        print(f"  After expansion: indices [{left_idx}:{right_idx+1}]")
                        print(f"  Values: [{filtered_data.iloc[left_idx]['values']:.3f}, "
                              f"{filtered_data.iloc[right_idx]['values']:.3f}]")
                    else:
                        print(f"\nPeak 3 (index {peak3_idx}, ALE={peak3['ale']:.6f}) NOT in range")
            else:
                print(f"\nPeak 2 (index {peak2_idx}, ALE={peak2['ale']:.6f}) NOT in range")
        
        # Store Range 1
        range_bins = filtered_data.iloc[left_idx:right_idx+1]
        ranges.append(create_range_info(range_bins, peaks_in_range, filtered_data, ale_values))
        print(f"\nRange 1 finalized: {len(peaks_in_range)} peak(s) included")
    
    # Range 2: Start from peak 2 if not already included
    if len(peak_info_list) >= 2 and peak_info_list[1]['index'] not in included_peaks:
        peak2 = peak_info_list[1]
        print(f"\n{'='*60}")
        print(f"RANGE 2: Starting from Peak 2 (index {peak2['index']}, ALE={peak2['ale']:.6f})")
        print('='*60)
        
        # Initial expansion ±2 bins
        left_idx, right_idx = expand_from_peak(ale_values, peak2['index'], n_bins_expand=2)
        peaks_in_range = [peak2['index']]
        included_peaks.add(peak2['index'])
        
        print(f"Initial expansion (±2): indices [{left_idx}:{right_idx+1}]")
        print(f"  Values: [{filtered_data.iloc[left_idx]['values']:.3f}, "
              f"{filtered_data.iloc[right_idx]['values']:.3f}]")
        
        # Check if peak 3 is included
        if len(peak_info_list) >= 3:
            peak3 = peak_info_list[2]
            peak3_idx = peak3['index']
            
            if left_idx <= peak3_idx <= right_idx:
                print(f"\nPeak 3 (index {peak3_idx}, ALE={peak3['ale']:.6f}) IS IN range")
                print(f"→ Expanding ±3 bins toward Peak 3")
                
                left_idx, right_idx = expand_toward_peak(ale_values, left_idx, right_idx,
                                                         peak3_idx, n_bins_expand=3)
                peaks_in_range.append(peak3_idx)
                included_peaks.add(peak3_idx)
                
                print(f"  After expansion: indices [{left_idx}:{right_idx+1}]")
                print(f"  Values: [{filtered_data.iloc[left_idx]['values']:.3f}, "
                      f"{filtered_data.iloc[right_idx]['values']:.3f}]")
            else:
                print(f"\nPeak 3 (index {peak3_idx}, ALE={peak3['ale']:.6f}) NOT in range")
        
        # Store Range 2
        range_bins = filtered_data.iloc[left_idx:right_idx+1]
        ranges.append(create_range_info(range_bins, peaks_in_range, filtered_data, ale_values))
        print(f"\nRange 2 finalized: {len(peaks_in_range)} peak(s) included")
    
    # Range 3: Start from peak 3 if not included and similar enough to peak 2
    if (len(peak_info_list) >= 3 and 
        peak_info_list[2]['index'] not in included_peaks and
        len(peak_info_list) >= 2):
        
        peak2 = peak_info_list[1]
        peak3 = peak_info_list[2]
        
        # Check similarity: peak3 ALE >= 80% of peak2 ALE
        if peak3['ale'] >= similarity_threshold * peak2['ale']:
            print(f"\n{'='*60}")
            print(f"RANGE 3: Starting from Peak 3 (index {peak3['index']}, ALE={peak3['ale']:.6f})")
            print(f"  Peak 3 ALE / Peak 2 ALE = {peak3['ale']/peak2['ale']:.2%} >= {similarity_threshold:.0%}")
            print('='*60)
            
            # Initial expansion ±2 bins (DO NOT check for peak 4)
            left_idx, right_idx = expand_from_peak(ale_values, peak3['index'], n_bins_expand=2)
            peaks_in_range = [peak3['index']]
            included_peaks.add(peak3['index'])
            
            print(f"Expansion (±2): indices [{left_idx}:{right_idx+1}]")
            print(f"  Values: [{filtered_data.iloc[left_idx]['values']:.3f}, "
                  f"{filtered_data.iloc[right_idx]['values']:.3f}]")
            print(f"  (Not checking for Peak 4)")
            
            # Store Range 3
            range_bins = filtered_data.iloc[left_idx:right_idx+1]
            ranges.append(create_range_info(range_bins, peaks_in_range, filtered_data, ale_values))
            print(f"\nRange 3 finalized: {len(peaks_in_range)} peak(s) included")
        else:
            print(f"\nPeak 3 ALE ({peak3['ale']:.6f}) < {similarity_threshold:.0%} of Peak 2 ALE "
                  f"({peak2['ale']:.6f})")
            print(f"  → Not creating Range 3")
    
    return ranges

def create_range_info(range_bins, peak_indices, filtered_data, ale_values):
    """Helper to create range info dictionary"""
    return {
        'min_value': range_bins['values'].min(),
        'max_value': range_bins['values'].max(),
        'peak_indices': peak_indices,
        'peak_values': [filtered_data.iloc[p]['values'] for p in peak_indices],
        'peak_ales': [ale_values[p] for p in peak_indices],
        'min_ale': range_bins['ale'].min(),
        'max_ale': range_bins['ale'].max(),
        'mean_ale': range_bins['ale'].mean(),
        'n_bins': len(range_bins),
        'n_peaks': len(peak_indices)
    }