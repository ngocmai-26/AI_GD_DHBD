"""
Data loading and preprocessing module
"""

import pandas as pd
import os
import numpy as np
from pathlib import Path
import config

def standardize_columns(df, mapping):
    """Standardize column names using mapping"""
    df_renamed = df.copy()
    for old_col, new_col in mapping.items():
        if old_col in df_renamed.columns:
            df_renamed.rename(columns={old_col: new_col}, inplace=True)
    return df_renamed

def read_all_excel_files(data_dir='dulieu/'):
    """Read all Excel files from data directory"""
    excel_files = []
    data_dir_path = Path(data_dir)
    
    for file_path in data_dir_path.glob('*.xlsx'):
        try:
            print(f"Reading {file_path.name}...")
            df = pd.read_excel(file_path)
            
            # Standardize columns
            df_std = standardize_columns(df, config.COLUMN_MAPPING)
            
            # Add source file name
            df_std['source_file'] = file_path.name
            
            excel_files.append({
                'name': file_path.name,
                'data': df_std
            })
            print(f"  - Loaded {len(df_std)} rows, {len(df_std.columns)} columns")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    return excel_files

def merge_to_enrollment(excel_files):
    """Merge all dataframes into enrollment table"""
    
    # Find the main enrollment data (DiemTong.xlsx)
    enrollment_df = None
    other_data = []
    
    for file_info in excel_files:
        df = file_info['data']
        if 'diemtong' in file_info['name'].lower():
            enrollment_df = df.copy()
            print(f"\nUsing {file_info['name']} as base enrollment table")
        else:
            other_data.append(file_info)
    
    # If DiemTong not found, use the largest dataframe
    if enrollment_df is None:
        largest_file = max(excel_files, key=lambda x: len(x['data']))
        enrollment_df = largest_file['data'].copy()
        print(f"\nUsing {largest_file['name']} as base enrollment table")
        other_data = [f for f in excel_files if f['name'] != largest_file['name']]
    
    # Ensure student_id exists and is numeric
    if 'student_id' in enrollment_df.columns:
        enrollment_df['student_id'] = pd.to_numeric(enrollment_df['student_id'], errors='coerce')
        enrollment_df = enrollment_df.dropna(subset=['student_id'])
    
    # Merge other dataframes
    merge_keys = ['student_id']
    if 'subject_id' in enrollment_df.columns:
        merge_keys.append('subject_id')
    if 'academic_year' in enrollment_df.columns:
        merge_keys.append('academic_year')
    if 'semester' in enrollment_df.columns:
        merge_keys.append('semester')
    
    print(f"\nMerging {len(other_data)} additional data sources...")
    
    for file_info in other_data:
        df = file_info['data']
        
        # Standardize merge keys in df
        available_keys = [k for k in merge_keys if k in df.columns]
        
        if available_keys:
            try:
                # Ensure data types match
                for key in available_keys:
                    if key == 'student_id':
                        df[key] = pd.to_numeric(df[key], errors='coerce')
                
                # Remove duplicates before merge
                df_unique = df.drop_duplicates(subset=available_keys)
                
                # Merge
                enrollment_df = enrollment_df.merge(
                    df_unique,
                    on=available_keys,
                    how='left',
                    suffixes=('', f'_{file_info["name"][:10]}')
                )
                print(f"  - Merged {file_info['name']}: {len(df_unique)} unique records")
            except Exception as e:
                print(f"  - Error merging {file_info['name']}: {e}")
    
    return enrollment_df

def create_pass_fail(enrollment_df):
    """Create pass_fail target variable"""
    df = enrollment_df.copy()
    
    # Try different columns to determine pass/fail
    if 'passed_module' in df.columns:
        df['pass_fail'] = df['passed_module'].apply(
            lambda x: 1 if pd.notna(x) and (str(x).strip().upper() in ['1', '1.0', 'TRUE', 'PASS', 'ĐẠT']) else 0
        )
    elif 'final_score' in df.columns or 'summary_score' in df.columns:
        score_col = 'final_score' if 'final_score' in df.columns else 'summary_score'
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
        df['pass_fail'] = df[score_col].apply(
            lambda x: 1 if pd.notna(x) and isinstance(x, (int, float)) and x >= 5.0 else 0
        )
    elif 'exam_score' in df.columns:
        # Convert to numeric, handling non-numeric values
        df['exam_score_numeric'] = pd.to_numeric(df['exam_score'], errors='coerce')
        df['pass_fail'] = df['exam_score_numeric'].apply(
            lambda x: 1 if pd.notna(x) and isinstance(x, (int, float)) and x >= 5.0 else 0
        )
    else:
        # Default: create binary from any score column
        score_cols = [col for col in df.columns if 'score' in col.lower() and col != 'pass_fail']
        if score_cols:
            score_col = score_cols[0]
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            df['pass_fail'] = df[score_col].apply(
                lambda x: 1 if pd.notna(x) and isinstance(x, (int, float)) and x >= 5.0 else 0
            )
        else:
            raise ValueError("Cannot create pass_fail: No suitable score column found")
    
    # Remove rows where pass_fail cannot be determined
    df = df.dropna(subset=['pass_fail'])
    
    print(f"\nCreated pass_fail column:")
    print(f"  - Pass (1): {df['pass_fail'].sum()}")
    print(f"  - Fail (0): {(df['pass_fail'] == 0).sum()}")
    
    return df

def preprocess_enrollment(enrollment_df):
    """Preprocess enrollment dataframe for modeling"""
    df = enrollment_df.copy()
    
    # Remove duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"\nRemoved {initial_len - len(df)} duplicate rows")
    
    # Select relevant features (numeric and categorical)
    # Exclude text columns, ID columns, and potential data leakage features
    exclude_cols = ['student_id', 'source_file', 'pass_fail', 
                    'full_name', 'last_name', 'first_name',
                    'subject_name', 'lecturer_name', 'major_name', 'faculty_name',
                    'reason_text', 'solution_text',
                    # Data leakage prevention - exclude features that directly determine target
                    'passed_module',  # This is used to create pass_fail target
                    'final_score',    # Direct score that determines pass/fail
                    'summary_score',  # Direct score that determines pass/fail
                    'exam_score',     # Direct exam score (though may be numeric string like 'VT')
                    'GC_registration_results',  # May contain pass/fail info
                    ]
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and not col.startswith('source_')]
    
    # Convert categorical columns to numeric
    for col in feature_cols:
        if df[col].dtype == 'object':
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    return df, feature_cols

