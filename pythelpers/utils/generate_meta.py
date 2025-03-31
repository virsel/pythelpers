#!/usr/bin/env python3
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import re
import pandas as pd
from datetime import datetime


def infer_column_type(column_name: str, values: List[Any], column_stats: Dict = None) -> str:
    """
    Infer the type of a column based on its values and name.
    
    Args:
        column_name: The name of the column
        values: Sample values from the column
        column_stats: Additional statistics about the column (optional)
    
    Returns:
        str: The inferred column type: 'continuous', 'categorical', 'ordinal', 'text', etc.
    """
    # Clean values by stripping whitespace and removing empty values
    cleaned_values = [v.strip() if isinstance(v, str) else v for v in values if v is not None]
    if not cleaned_values:
        return "categorical"  # Default if no valid values
    
    # Check for ID columns first
    id_indicators = ['id', 'key', 'code', 'uuid', 'guid', 'identifier']
    if column_name.lower() == 'id' or any(column_name.lower().endswith(f"_{ind}") or column_name.lower().endswith(f"{ind}") for ind in id_indicators):
        return "categorical"
    
    # Check for boolean columns
    unique_strings = set(str(v).lower() for v in cleaned_values if v is not None)
    boolean_sets = [
        {'true', 'false'},
        {'t', 'f'},
        {'yes', 'no'},
        {'y', 'n'},
        {'1', '0'},
        {'1.0', '0.0'}
    ]
    
    for boolean_set in boolean_sets:
        if unique_strings.issubset(boolean_set) and len(unique_strings) <= 2:
            return "categorical"
    
    # Check for text/tokens
    if column_name.lower().endswith('_tks') or 'token' in column_name.lower() or 'text' in column_name.lower():
        # If the average string length is large, assume it's text
        if column_stats and 'mean_length' in column_stats and column_stats['mean_length'] > 20:
            return "text"
        # Check if values typically contain multiple words
        text_indicators = 0
        for val in cleaned_values:
            if isinstance(val, str) and len(val.split()) > 3:  # More than 3 words
                text_indicators += 1
        
        if text_indicators > len(cleaned_values) * 0.3:  # If 30% of values appear to be text
            return "text"
    
    # Handle category column names
    if 'category' in column_name.lower() or 'cat_' in column_name.lower() or column_name.lower().startswith('cat') or column_name.lower().endswith('_cat'):
        return "categorical"
    
    # Try to convert to numeric
    numeric_values = []
    numeric_count = 0
    for val in cleaned_values:
        try:
            if isinstance(val, str):
                # Remove commas and other formatting that might be in numbers
                cleaned = re.sub(r'[,$%]', '', val.strip())
                if cleaned:
                    num_val = float(cleaned)
                    numeric_values.append(num_val)
                    numeric_count += 1
            elif isinstance(val, (int, float)):
                numeric_values.append(float(val))
                numeric_count += 1
        except (ValueError, TypeError):
            pass
    
    # If more than 70% of non-empty values are numeric
    if numeric_count >= 0.7 * len(cleaned_values) and len(cleaned_values) > 0:
        # Check if the values are all integers or have very few unique values
        # compared to the total (suggesting categorical)
        if numeric_values:
            unique_values = set(numeric_values)
            # Check if all values are integers
            all_ints = all(float(v).is_integer() for v in numeric_values)
            
            # If all integers and few unique values, might be categorical
            if all_ints and len(unique_values) <= 10 and len(unique_values) < 0.2 * len(numeric_values):
                return "categorical"
            
            # If small number of unique values, might be ordinal
            if len(unique_values) <= 20 and len(unique_values) < 0.3 * len(numeric_values):
                return "ordinal"
            
            return "continuous"
    
    # Check for date patterns
    date_count = 0
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',                   # ISO format: 2023-01-15
        r'\d{2}/\d{2}/\d{4}',                   # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',                   # MM-DD-YYYY
        r'\w+\s+\d{1,2},\s+\d{4}'               # Month DD, YYYY
    ]
    
    for val in cleaned_values:
        if isinstance(val, str):
            for pattern in date_patterns:
                if re.search(pattern, val):
                    date_count += 1
                    break
    
    if date_count >= 0.7 * len(cleaned_values):
        return "datetime"
    
    # Count unique values
    unique_values = set(str(v) for v in cleaned_values)
    
    # If very few unique values compared to total, likely categorical
    if len(unique_values) <= 10 or len(unique_values) < 0.1 * len(cleaned_values):
        return "categorical"
    
    # If moderate number of unique values, might be ordinal
    if len(unique_values) <= 30 or len(unique_values) < 0.3 * len(cleaned_values):
        # Names with indicators of ordinal nature
        ordinal_indicators = ['grade', 'level', 'tier', 'class', 'rank', 'stage', 'step', 'education', 'year', 'rating']
        for indicator in ordinal_indicators:
            if indicator in column_name.lower():
                return "ordinal"
    
    # Check for long text values
    long_text_count = 0
    for val in cleaned_values:
        if isinstance(val, str) and len(val) > 100:  # Long text
            long_text_count += 1
    
    if long_text_count > len(cleaned_values) * 0.2:  # If 20% are long text
        return "text"
    
    # Default to categorical for other scenarios
    return "categorical"


def generate_metadata(csv_file_path: Path, output_json_path: Path, sample_size: int = None) -> None:
    """
    Generate a metadata JSON file from a CSV file.
    
    Args:
        csv_file_path: Path to the input CSV file
        output_json_path: Path to save the output JSON metadata
        sample_size: Number of rows to sample for type inference (None = all rows)
    """
    # Convert to Path objects if they're not already
    csv_file_path = Path(csv_file_path)
    output_json_path = Path(output_json_path)
    
    # Check if the file exists
    if not csv_file_path.exists():
        print(f"Error: File '{csv_file_path}' not found.")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Absolute path attempted: {csv_file_path.absolute()}")
        return
    
    try:
        # Try using pandas for more robust CSV handling
        # If sample_size is None, read the entire file
        print(f"Reading CSV file: {csv_file_path}")
        if sample_size is None:
            df = pd.read_csv(csv_file_path)
        else:
            df = pd.read_csv(csv_file_path, nrows=sample_size)
            
        headers = df.columns.tolist()
        
        # Create metadata structure
        metadata = {
            "filename": str(csv_file_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows_analyzed": len(df),
            "total_rows": len(df),
            "columns": []
        }
        
        for header in headers:
            # Get non-null values for the column
            values = df[header].dropna().tolist()
            
            # Collect additional stats to help with type inference
            column_stats = {}
            
            # Calculate string length stats for text detection
            if df[header].dtype == 'object':
                # Get mean string length for text columns
                try:
                    str_lengths = df[header].astype(str).apply(len)
                    column_stats['mean_length'] = str_lengths.mean()
                    column_stats['max_length'] = str_lengths.max()
                except:
                    pass
            
            column_type = infer_column_type(header, values, column_stats)
            
            # Manually override types for specific column patterns
            if header == 'id' or header.endswith('_id'):
                column_type = "categorical"
            
            if header.startswith('category_') or 'category' in header.lower():
                column_type = "categorical"
            
            if 'combined_tks' in header or header.endswith('_tks'):
                column_type = "text"
            
            column_meta = {
                "name": header,
                "type": column_type
            }
            
            # Add additional statistics based on column type
            if column_type == "continuous":
                column_meta.update({
                    "min": float(df[header].min()) if not df[header].empty else None,
                    "max": float(df[header].max()) if not df[header].empty else None,
                    "mean": float(df[header].mean()) if not df[header].empty else None,
                    "null_count": int(df[header].isna().sum())
                })
            elif column_type in ["categorical", "ordinal"]:
                # Get value counts for the most common values
                value_counts = df[header].value_counts().head(10).to_dict()
                column_meta.update({
                    "unique_values": len(df[header].unique()),
                    "most_common": value_counts,
                    "null_count": int(df[header].isna().sum())
                })
            elif column_type == "text":
                column_meta.update({
                    "avg_length": column_stats.get('mean_length', 0),
                    "max_length": column_stats.get('max_length', 0),
                    "null_count": int(df[header].isna().sum())
                })
            
            metadata["columns"].append(column_meta)
        
    except Exception as e:
        print(f"Error using pandas: {str(e)}")
        print("Falling back to basic CSV reader...")
        
        try:
            # Fall back to basic CSV reader
            with open(csv_file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
                # Read the header
                reader = csv.reader(csvfile)
                headers = next(reader)
                headers = [h.strip() for h in headers]
                
                # Read all rows if sample_size is None
                all_rows = list(reader) if sample_size is None else []
                
                # If we're sampling, just get those rows
                if sample_size is not None and not all_rows:
                    for i, row in enumerate(reader):
                        if i >= sample_size:
                            break
                        all_rows.append(row)
                
                # Sample the data for type inference
                sample_data = {header: [] for header in headers}
                for row in all_rows:
                    for j, header in enumerate(headers):
                        if j < len(row):
                            sample_data[header].append(row[j])
            
            # Create basic metadata structure
            metadata = {
                "filename": str(csv_file_path),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rows_analyzed": len(all_rows),
                "total_rows": len(all_rows),
                "columns": []
            }
            
            for header in headers:
                # Manually override types for specific column patterns
                if header == 'id' or header.endswith('_id'):
                    column_type = "categorical"
                elif header.startswith('category_') or 'category' in header.lower():
                    column_type = "categorical"
                elif 'combined_tks' in header or header.endswith('_tks'):
                    column_type = "text"
                else:
                    column_type = infer_column_type(header, sample_data[header])
                
                metadata["columns"].append({
                    "name": header,
                    "type": column_type
                })
        except Exception as e:
            print(f"Error with basic CSV reader: {str(e)}")
            print("Failed to process the CSV file.")
            return
    
    try:
        # Create output directory if it doesn't exist
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If we want to use "text" as a type, uncomment the following code
        # For the simplified output, map "text" to "categorical" to match the desired format
        simplified = {
            "columns": [
                {
                    "name": col["name"],
                    # Map "text" type to "categorical" if needed to match the requested format
                    # "type": "categorical" if col["type"] == "text" else col["type"]
                    "type": col["type"]
                }
                for col in metadata["columns"]
            ]
        }
        
        # Write metadata to JSON file
        with open(output_json_path, 'w') as jsonfile:
            json.dump(simplified, jsonfile, indent=4)
        
        print(f"Metadata file successfully generated at {output_json_path}")
        print(f"Analyzed {metadata['rows_analyzed']} rows from {csv_file_path}")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata JSON from any CSV file")
    parser.add_argument("--csv_file", default="data.csv", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", default="meta.json", help="Path to output JSON file")
    parser.add_argument("--sample", "-s", type=int, help="Number of rows to sample (default: all rows)")
    parser.add_argument("--full", "-f", action="store_true", help="Include full statistics in output")
    
    args = parser.parse_args()

    # Get the base directory (where the script is located)
    base_dir = Path(__file__).parent
    
    # Set file paths relative to the script location
    csv_file_path = base_dir / 'features4ausw4linearsvc_train.csv'
    output_path = base_dir / 'features4ausw4linearsvc_train.json'
    
    # Print paths for debugging
    print(f"Script location: {base_dir}")
    print(f"CSV path: {csv_file_path}")
    print(f"Output path: {output_path}")
    
    # Use either hardcoded paths or args
    # Uncomment the following line to use command line arguments
    # csv_file_path = Path(args.csv_file)
    # output_path = Path(args.output)
    
    generate_metadata(
        csv_file_path=csv_file_path,
        output_json_path=output_path,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()