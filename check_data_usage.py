"""
Script to check if all data is being used in the pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config
from data_loader import read_all_excel_files, merge_to_enrollment, create_pass_fail, preprocess_enrollment
from model_trainer import prepare_features

def check_data_usage():
    """Check data usage at each step"""
    
    print("="*80)
    print("KIỂM TRA SỬ DỤNG DỮ LIỆU")
    print("="*80)
    
    # Step 1: Read all Excel files
    print("\n[Step 1] Đọc Excel files...")
    excel_files = read_all_excel_files('dulieu/')
    total_rows_from_files = sum(len(f['data']) for f in excel_files)
    print(f"  Tổng số dòng từ tất cả file: {total_rows_from_files:,}")
    
    # Step 2: Merge to enrollment
    print("\n[Step 2] Merge về enrollment table...")
    enrollment_df = merge_to_enrollment(excel_files)
    rows_after_merge = len(enrollment_df)
    print(f"  Số dòng sau merge: {rows_after_merge:,}")
    print(f"  Mất: {total_rows_from_files - rows_after_merge:,} dòng (do merge)")
    
    # Check student_id
    if 'student_id' in enrollment_df.columns:
        missing_student_id = enrollment_df['student_id'].isna().sum()
        print(f"  Dòng thiếu student_id: {missing_student_id:,}")
        if missing_student_id > 0:
            print(f"  ⚠️  CẢNH BÁO: {missing_student_id:,} dòng sẽ bị loại bỏ ở bước merge!")
    
    # Step 3: Create pass_fail
    print("\n[Step 3] Tạo pass_fail target...")
    enrollment_df_before_passfail = enrollment_df.copy()
    enrollment_df = create_pass_fail(enrollment_df)
    rows_after_passfail = len(enrollment_df)
    rows_lost_passfail = len(enrollment_df_before_passfail) - rows_after_passfail
    print(f"  Số dòng trước khi tạo pass_fail: {len(enrollment_df_before_passfail):,}")
    print(f"  Số dòng sau khi tạo pass_fail: {rows_after_passfail:,}")
    print(f"  Mất: {rows_lost_passfail:,} dòng (không tạo được pass_fail)")
    
    if rows_lost_passfail > 0:
        print(f"  ⚠️  CẢNH BÁO: {rows_lost_passfail:,} dòng bị loại bỏ!")
        # Check why
        temp_df = enrollment_df_before_passfail.copy()
        if 'passed_module' in temp_df.columns:
            missing_passed = temp_df['passed_module'].isna().sum()
            print(f"    - Dòng thiếu passed_module: {missing_passed:,}")
        if 'final_score' in temp_df.columns:
            missing_final = temp_df['final_score'].isna().sum()
            print(f"    - Dòng thiếu final_score: {missing_final:,}")
        if 'summary_score' in temp_df.columns:
            missing_summary = temp_df['summary_score'].isna().sum()
            print(f"    - Dòng thiếu summary_score: {missing_summary:,}")
    
    # Step 4: Preprocess
    print("\n[Step 4] Preprocessing...")
    enrollment_df_before_preprocess = enrollment_df.copy()
    enrollment_df_processed, feature_cols = preprocess_enrollment(enrollment_df)
    rows_after_preprocess = len(enrollment_df_processed)
    rows_lost_preprocess = len(enrollment_df_before_preprocess) - rows_after_preprocess
    print(f"  Số dòng trước preprocessing: {len(enrollment_df_before_preprocess):,}")
    print(f"  Số dòng sau preprocessing: {rows_after_preprocess:,}")
    print(f"  Mất: {rows_lost_preprocess:,} dòng (duplicates)")
    
    # Step 5: Prepare features
    print("\n[Step 5] Prepare features...")
    X, y, feature_names = prepare_features(enrollment_df_processed, feature_cols)
    rows_after_features = len(X)
    rows_lost_features = len(enrollment_df_processed) - rows_after_features
    print(f"  Số dòng trước prepare_features: {len(enrollment_df_processed):,}")
    print(f"  Số dòng sau prepare_features: {rows_after_features:,}")
    print(f"  Mất: {rows_lost_features:,} dòng (NaN target)")
    
    if rows_lost_features > 0:
        print(f"  ⚠️  CẢNH BÁO: {rows_lost_features:,} dòng bị loại bỏ do thiếu target!")
    
    # Summary
    print("\n" + "="*80)
    print("TỔNG KẾT")
    print("="*80)
    print(f"Tổng dòng ban đầu: {total_rows_from_files:,}")
    print(f"Dòng cuối cùng sử dụng: {rows_after_features:,}")
    print(f"Tổng số dòng bị mất: {total_rows_from_files - rows_after_features:,}")
    print(f"Tỷ lệ sử dụng: {rows_after_features/total_rows_from_files*100:.2f}%")
    print(f"Tỷ lệ mất: {(total_rows_from_files - rows_after_features)/total_rows_from_files*100:.2f}%")
    
    # Breakdown by step
    print("\nPhân tích từng bước:")
    print(f"  1. Sau merge: {rows_after_merge:,} ({rows_after_merge/total_rows_from_files*100:.2f}%)")
    print(f"  2. Sau pass_fail: {rows_after_passfail:,} ({rows_after_passfail/total_rows_from_files*100:.2f}%)")
    print(f"  3. Sau preprocessing: {rows_after_preprocess:,} ({rows_after_preprocess/total_rows_from_files*100:.2f}%)")
    print(f"  4. Sau prepare_features: {rows_after_features:,} ({rows_after_features/total_rows_from_files*100:.2f}%)")
    
    # Check for potential data loss issues
    print("\n" + "="*80)
    print("ĐÁNH GIÁ")
    print("="*80)
    
    if rows_after_features / total_rows_from_files < 0.8:
        print("⚠️  CẢNH BÁO: Chỉ sử dụng < 80% dữ liệu ban đầu!")
        print("   Cần kiểm tra lại các bước loại bỏ dữ liệu.")
    elif rows_after_features / total_rows_from_files < 0.9:
        print("⚠️  Lưu ý: Sử dụng < 90% dữ liệu ban đầu.")
        print("   Có thể có cơ hội cải thiện bằng cách xử lý tốt hơn missing values.")
    else:
        print("✓ Tốt: Sử dụng > 90% dữ liệu ban đầu.")
    
    # Detailed breakdown
    if rows_lost_passfail > 0:
        print(f"\n⚠️  {rows_lost_passfail:,} dòng bị mất khi tạo pass_fail.")
        print("   Khuyến nghị: Kiểm tra lại logic tạo pass_fail để giữ lại nhiều dữ liệu hơn.")
    
    if rows_lost_features > 0:
        print(f"\n⚠️  {rows_lost_features:,} dòng bị mất do thiếu target.")
        print("   Khuyến nghị: Đảm bảo tất cả dòng đều có pass_fail được tạo.")
    
    return {
        'total_rows_from_files': total_rows_from_files,
        'rows_after_merge': rows_after_merge,
        'rows_after_passfail': rows_after_passfail,
        'rows_after_preprocess': rows_after_preprocess,
        'rows_after_features': rows_after_features,
        'usage_percentage': rows_after_features / total_rows_from_files * 100
    }

if __name__ == "__main__":
    check_data_usage()



