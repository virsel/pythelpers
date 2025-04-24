import pandas as pd
import numpy as np

def auto_convert_numeric_strings(df):
    """Automatically detect and convert string columns that contain numbers"""
    df_result = df.copy()
    
    # Get string columns
    string_columns = df.select_dtypes(include=['object']).columns
    
    for col in string_columns:
        # Check if column contains numeric strings
        # Sample a few values to test
        sample = df[col].dropna().head(10)
        try:
            # Try converting sample to see if it works
            pd.to_numeric(sample)
            # If it works, convert the whole column
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce')
            print(f"Converted column {col} to numeric")
        except:
            # Keep as string if conversion fails
            print(f"Keeping column {col} as string")
    
    return df_result

def statistic_similarity(df1, df2):
    # make sure all numeric
    df2 = auto_convert_numeric_strings(df2)
    df1 = auto_convert_numeric_strings(df1)
    
    # Statistical similarity
    real_means = df2.mean()
    gen_means = df1.mean()
    mean_diff = np.mean(np.abs(real_means - gen_means))
    
    # Column correlations with proper error handling
    try:
        # Drop constant columns which cause NaN in correlation
        real_data_corr = df2.loc[:, df2.nunique() > 1]
        generated_data_corr = df1.loc[:, df1.nunique() > 1]
        
        # Make sure we use the same columns for both dataframes
        common_cols = list(set(real_data_corr.columns) & set(generated_data_corr.columns))
        if len(common_cols) < 2:  # Need at least 2 columns for correlation
            corr_diff = np.nan
        else:
            real_corr = real_data_corr[common_cols].corr().fillna(0).values
            gen_corr = generated_data_corr[common_cols].corr().fillna(0).values
            corr_diff = np.mean(np.abs(real_corr - gen_corr))
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        corr_diff = np.nan
    
    # Distribution check (for continuous columns)
    dist_diff = 0
    continuous_cols = df2.select_dtypes(include=['float64', 'int64']).columns
    if len(continuous_cols) > 0:
        for col in continuous_cols:
            try:
                real_hist = np.histogram(df2[col], bins=20, density=True)[0]
                gen_hist = np.histogram(df1[col], bins=20, density=True)[0]
                dist_diff += np.sum(np.abs(real_hist - gen_hist)) / len(real_hist)
            except Exception as e:
                print(f"Error calculating histogram for column {col}: {e}")
        dist_diff = dist_diff / len(continuous_cols)  # Average over all columns
    else:
        dist_diff = np.nan
    
    # Average only non-NaN metrics
    valid_metrics = [x for x in [mean_diff, corr_diff, dist_diff] if not np.isnan(x)]
    if valid_metrics:
        avg_metric = sum(valid_metrics) / len(valid_metrics)
    else:
        avg_metric = np.nan
        
    return mean_diff, corr_diff, dist_diff, avg_metric