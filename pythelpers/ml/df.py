import pandas as pd

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