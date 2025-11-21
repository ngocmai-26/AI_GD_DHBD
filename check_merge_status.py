"""
Check merge status of all data sources
"""

import pandas as pd
import os

def check_merge_status():
    """Check what data has been merged"""
    
    print("="*80)
    print("KIỂM TRA TRẠNG THÁI MERGE CÁC NGUỒN DỮ LIỆU")
    print("="*80)
    
    # Read processed enrollment data
    df = pd.read_csv('output/data/enrollment_processed.csv')
    
    print(f"\nTổng số dòng: {len(df):,}")
    print(f"Tổng số cột: {len(df.columns)}")
    
    # Check each data source
    sources = {
        'Điểm rèn luyện (diemrenluyen.xlsx)': {
            'keywords': ['renluyen', 'conduct', 'diemrenluyen'],
            'expected_cols': ['conduct_score', 'conduct_classification']
        },
        'Điểm danh (Điểm danh Khoa FIRA.xlsx)': {
            'keywords': ['diemdanh', 'attendance', 'danh'],
            'expected_cols': ['attendance_date', 'attendance_status']
        },
        'Nhân khẩu (nhankhau.xlsx)': {
            'keywords': ['nhankhau', 'address', 'birth', 'gender', 'gioitinh', 'ngaysinh', 'diachi'],
            'expected_cols': ['Birthdate', 'Gender', 'place_of_birth', 'addressCMND']
        },
        'Tự học (tuhoc.xlsx)': {
            'keywords': ['tuhoc', 'study', 'self_study'],
            'expected_cols': ['study_time', 'study_hours', 'study_minutes']
        },
        'DS Điểm (dsdiem.xlsx)': {
            'keywords': ['dsdiem'],
            'expected_cols': []
        },
        'PPDG.xlsx': {
            'keywords': ['PPDG', 'ppdg'],
            'expected_cols': []
        },
        'PPGD.xlsx': {
            'keywords': ['PPGD', 'ppgd'],
            'expected_cols': []
        }
    }
    
    print("\n" + "="*80)
    print("CHI TIẾT TỪNG NGUỒN DỮ LIỆU")
    print("="*80)
    
    for source_name, config in sources.items():
        print(f"\n{source_name}:")
        print("-" * 80)
        
        # Find columns from this source
        matching_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in config['keywords']):
                matching_cols.append(col)
        
        if matching_cols:
            print(f"  ✓ Đã được merge: {len(matching_cols)} cột")
            print(f"  - Các cột: {matching_cols[:10]}")
            
            # Check coverage
            total_with_data = 0
            for col in matching_cols[:5]:
                non_null = df[col].notna().sum()
                coverage = non_null / len(df) * 100
                print(f"    • {col}: {non_null:,}/{len(df):,} dòng ({coverage:.1f}%)")
                if coverage > total_with_data:
                    total_with_data = coverage
            
            # Check if expected columns exist
            if config['expected_cols']:
                missing = [col for col in config['expected_cols'] if col not in matching_cols]
                if missing:
                    print(f"  ⚠️  Thiếu cột: {missing}")
                else:
                    print(f"  ✓ Có đủ các cột mong đợi")
            
            # Calculate overall coverage
            if matching_cols:
                # Use the column with best coverage
                best_col = max(matching_cols, key=lambda x: df[x].notna().sum())
                best_coverage = df[best_col].notna().sum() / len(df) * 100
                print(f"  - Tỷ lệ coverage tốt nhất: {best_coverage:.1f}%")
                
                if best_coverage < 50:
                    print(f"  ⚠️  CẢNH BÁO: Coverage thấp (<50%)")
                elif best_coverage < 80:
                    print(f"  ⚠️  Lưu ý: Coverage trung bình (50-80%)")
                else:
                    print(f"  ✓ Coverage tốt (>80%)")
        else:
            print(f"  ✗ CHƯA ĐƯỢC MERGE hoặc không có cột nào khớp")
    
    # Summary
    print("\n" + "="*80)
    print("TỔNG KẾT")
    print("="*80)
    
    all_sources_merged = True
    for source_name, config in sources.items():
        matching_cols = [col for col in df.columns 
                        if any(keyword in col.lower() for keyword in config['keywords'])]
        if not matching_cols:
            print(f"✗ {source_name}: CHƯA MERGE")
            all_sources_merged = False
        else:
            best_col = max(matching_cols, key=lambda x: df[x].notna().sum())
            coverage = df[best_col].notna().sum() / len(df) * 100
            status = "✓" if coverage > 50 else "⚠️"
            print(f"{status} {source_name}: Đã merge ({coverage:.1f}% coverage)")
    
    if all_sources_merged:
        print("\n✓ TẤT CẢ các nguồn dữ liệu đã được merge vào enrollment table")
    else:
        print("\n⚠️  Một số nguồn dữ liệu chưa được merge hoặc coverage thấp")
    
    # Explain why coverage might be low
    print("\n" + "="*80)
    print("GIẢI THÍCH VỀ COVERAGE")
    print("="*80)
    print("\nCoverage không đạt 100% là BÌNH THƯỜNG vì:")
    print("  • Merge dựa trên student_id (và subject_id, academic_year, semester)")
    print("  • Không phải tất cả enrollment records đều có dữ liệu từ mọi nguồn")
    print("  • Ví dụ: Điểm danh chỉ có 13% vì không phải tất cả môn học đều có điểm danh")
    print("  • Điểm rèn luyện có 56% vì chỉ tính theo học kỳ, không phải theo môn học")
    print("\n✓ Điều quan trọng là DỮ LIỆU ĐÃ ĐƯỢC MERGE, mô hình sẽ sử dụng")
    print("  những dòng có dữ liệu đó để học các patterns quan trọng")

if __name__ == "__main__":
    check_merge_status()



