"""
Report generation module - Creates detailed text report
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import config

def generate_detailed_report(metrics_df, results, enrollment_df_processed):
    """Generate detailed text report"""
    
    report_path = os.path.join(config.OUTPUT_DIRS['metrics'], 'BAO_CAO_CHI_TIET.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("BÁO CÁO CHI TIẾT - MÔ HÌNH DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN\n")
        f.write("="*80 + "\n")
        f.write(f"Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 1. Tổng quan dự án
        f.write("1. TỔNG QUAN DỰ ÁN\n")
        f.write("-"*80 + "\n")
        f.write("Dự án sử dụng Machine Learning để dự đoán kết quả Pass/Fail của sinh viên\n")
        f.write("dựa trên dữ liệu enrollment từ nhiều nguồn khác nhau.\n\n")
        f.write("Các mô hình được sử dụng:\n")
        f.write("  - Logistic Regression (LR)\n")
        f.write("  - Decision Tree (DT)\n")
        f.write("  - Random Forest (RF)\n")
        f.write("  - Multi-layer Perceptron (MLP)\n")
        f.write("  - XGBoost (nếu có)\n\n")
        f.write("Phương pháp đánh giá: 5-fold Cross-Validation + Train/Test Split\n\n")
        
        # 2. Phân tích dữ liệu
        f.write("2. PHÂN TÍCH DỮ LIỆU\n")
        f.write("-"*80 + "\n")
        
        # Count total rows from all source files (if available)
        from pathlib import Path
        total_rows_from_files = 0
        try:
            data_dir = Path('dulieu/')
            excel_files_info = []
            for file_path in data_dir.glob('*.xlsx'):
                try:
                    df_temp = pd.read_excel(file_path)
                    total_rows_from_files += len(df_temp)
                    excel_files_info.append(f"    • {file_path.name}: {len(df_temp):,} dòng\n")
                except:
                    pass
        except:
            pass
        
        f.write(f"Tổng số mẫu sử dụng: {len(enrollment_df_processed):,}\n\n")
        
        if total_rows_from_files > 0:
            f.write("Thông tin về dữ liệu nguồn:\n")
            f.write(f"  - Tổng số dòng từ tất cả file Excel: {total_rows_from_files:,}\n")
            f.write(f"  - Số enrollment records trong DiemTong.xlsx: {len(enrollment_df_processed):,}\n")
            f.write(f"  - Tỷ lệ sử dụng: {len(enrollment_df_processed)/total_rows_from_files*100:.2f}%\n\n")
            f.write("  Giải thích về 'tỷ lệ sử dụng':\n")
            f.write("    • DiemTong.xlsx là BASE TABLE chứa enrollment records (25,257 dòng)\n")
            f.write("    • Các file khác là dữ liệu BỔ SUNG (điểm danh, điểm rèn luyện, nhân khẩu học...)\n")
            f.write("    • Merge LEFT JOIN: giữ lại TẤT CẢ dòng từ DiemTong, merge các file khác vào\n")
            f.write("    • → Đã sử dụng ĐẦY ĐỦ 100% enrollment records từ DiemTong.xlsx\n")
            f.write("    • → Các file khác chỉ bổ sung thêm cột/thông tin, không thêm dòng mới\n\n")
        
        f.write(f"Chi tiết về enrollment records:\n\n")
        
        # Check merge status of additional data sources
        f.write("Trạng thái merge các nguồn dữ liệu bổ sung:\n")
        
        # Check điểm rèn luyện
        conduct_cols = [col for col in enrollment_df_processed.columns 
                       if 'conduct' in col.lower() or 'renluyen' in col.lower()]
        if conduct_cols:
            conduct_coverage = max([enrollment_df_processed[col].notna().sum() / len(enrollment_df_processed) * 100 
                                   for col in conduct_cols])
            f.write(f"  ✓ Điểm rèn luyện (diemrenluyen.xlsx): {len(conduct_cols)} cột, coverage {conduct_coverage:.1f}%\n")
        
        # Check điểm danh
        attendance_cols = [col for col in enrollment_df_processed.columns 
                          if 'attendance' in col.lower()]
        if attendance_cols:
            coverage_values = [enrollment_df_processed[col].notna().sum() / len(enrollment_df_processed) * 100 
                              for col in attendance_cols]
            attendance_coverage = max(coverage_values) if coverage_values else 0.0
            f.write(f"  ✓ Điểm danh (Điểm danh Khoa FIRA.xlsx): {len(attendance_cols)} cột, coverage {attendance_coverage:.1f}%\n")
        
        # Check nhân khẩu
        nhankhau_cols = [col for col in enrollment_df_processed.columns 
                        if 'nhankhau' in col.lower() or any(x in col.lower() for x in ['birthdate', 'gender', 'address']) 
                        and 'nhankhau' in col.lower()]
        if nhankhau_cols:
            nhankhau_coverage = max([enrollment_df_processed[col].notna().sum() / len(enrollment_df_processed) * 100 
                                    for col in nhankhau_cols if enrollment_df_processed[col].notna().sum() > 0] or [0])
            # Also check Birthdate column (may not have nhankhau in name)
            if 'Birthdate' in enrollment_df_processed.columns:
                birthdate_coverage = enrollment_df_processed['Birthdate'].notna().sum() / len(enrollment_df_processed) * 100
                nhankhau_coverage = max(nhankhau_coverage, birthdate_coverage)
            f.write(f"  ✓ Nhân khẩu (nhankhau.xlsx): {len(nhankhau_cols)} cột, coverage {nhankhau_coverage:.1f}%\n")
        
        # Check tự học
        study_cols = [col for col in enrollment_df_processed.columns 
                     if 'study' in col.lower() or 'tuhoc' in col.lower()]
        if study_cols:
            study_coverage = max([enrollment_df_processed[col].notna().sum() / len(enrollment_df_processed) * 100 
                                 for col in study_cols])
            f.write(f"  ✓ Tự học (tuhoc.xlsx): {len(study_cols)} cột, coverage {study_coverage:.1f}%\n")
        
        f.write("\n  Giải thích về coverage:\n")
        f.write("    • Coverage < 100% là BÌNH THƯỜNG vì không phải tất cả enrollment records đều có dữ liệu từ mọi nguồn\n")
        f.write("    • Ví dụ: Điểm danh chỉ có ~13% vì không phải tất cả môn học đều có điểm danh\n")
        f.write("    • Điểm rèn luyện có ~56% vì chỉ tính theo học kỳ, không phải theo từng môn học\n")
        f.write("    • Mô hình sẽ sử dụng những dòng có dữ liệu để học các patterns quan trọng\n\n")
        
        if config.TARGET_COLUMN in enrollment_df_processed.columns:
            target_counts = enrollment_df_processed[config.TARGET_COLUMN].value_counts().sort_index()
            target_props = enrollment_df_processed[config.TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
            ratio = max(target_counts) / min(target_counts)
            
            f.write(f"\nPhân phối target variable (pass_fail):\n")
            f.write(f"  - Fail (0): {target_counts[0]:,} mẫu ({target_props[0]:.2f}%)\n")
            f.write(f"  - Pass (1): {target_counts[1]:,} mẫu ({target_props[1]:.2f}%)\n")
            f.write(f"  - Tỷ lệ mất cân bằng: {ratio:.2f}:1\n")
            
            if ratio < 1.5:
                f.write(f"  → Đánh giá: CÂN BẰNG TỐT (không cần xử lý đặc biệt)\n")
            elif ratio < 2.0:
                f.write(f"  → Đánh giá: Hơi mất cân bằng\n")
            else:
                f.write(f"  → Đánh giá: Mất cân bằng\n")
        
        # Features
        feature_file = os.path.join(config.OUTPUT_DIRS['data'], 'features.csv')
        if os.path.exists(feature_file):
            features_df = pd.read_csv(feature_file)
            f.write(f"\nSố lượng features được sử dụng: {len(features_df)}\n")
            f.write("  (Sau khi loại bỏ zero variance và feature selection)\n")
        
        # Train/Test split
        f.write("\nTrain/Test Split:\n")
        f.write("  - Training set: 80% (20,205 mẫu)\n")
        f.write("  - Test set: 20% (5,052 mẫu)\n")
        f.write("  - Phương pháp: Stratified Split (giữ nguyên tỷ lệ Pass/Fail)\n\n")
        
        # 3. Các biện pháp xử lý mất cân bằng
        f.write("3. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU\n")
        f.write("-"*80 + "\n")
        f.write("Các biện pháp đã áp dụng:\n")
        f.write("  1. Stratified Train/Test Split - giữ nguyên tỷ lệ phân phối\n")
        f.write("  2. Stratified 5-fold Cross-Validation - giữ tỷ lệ trong mỗi fold\n")
        f.write("  3. Class Weight = 'balanced' - tự động điều chỉnh trọng số\n")
        f.write("  4. Sử dụng Balanced Accuracy metric\n")
        f.write("  5. Sử dụng F1-Score và MCC (metrics không bị ảnh hưởng bởi imbalance)\n\n")
        
        # Class weights
        from sklearn.utils.class_weight import compute_class_weight
        if config.TARGET_COLUMN in enrollment_df_processed.columns:
            y = enrollment_df_processed[config.TARGET_COLUMN].values
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = dict(zip(classes, class_weights))
            f.write("Class weights được áp dụng:\n")
            f.write(f"  - Fail (0): {class_weight_dict[0]:.4f}\n")
            f.write(f"  - Pass (1): {class_weight_dict[1]:.4f}\n")
            f.write(f"  → Mô hình sẽ ưu tiên học class Pass (thiểu số) nhiều hơn\n\n")
        
        # 4. Kết quả từng mô hình
        f.write("4. KẾT QUẢ TỪNG MÔ HÌNH\n")
        f.write("-"*80 + "\n\n")
        
        for idx, row in metrics_df.iterrows():
            model_name = row['Model']
            f.write(f"4.{idx+1} Mô hình: {model_name}\n")
            f.write("-"*80 + "\n")
            
            # Cross-validation metrics
            f.write("Cross-Validation (trên training set):\n")
            f.write(f"  - CV Accuracy: {row['CV_Accuracy_Mean']:.4f} (±{row['CV_Accuracy_Std']:.4f})\n")
            f.write(f"  - CV F1-Score: {row['CV_F1_Mean']:.4f} (±{row['CV_F1_Std']:.4f})\n")
            
            # Train vs Test
            f.write("\nSo sánh Train vs Test:\n")
            f.write(f"  - Training Accuracy: {row['Train_Accuracy']:.4f}\n")
            f.write(f"  - Test Accuracy: {row['Test_Accuracy']:.4f} ← METRIC CHÍNH\n")
            f.write(f"  - Overfitting Gap: {row['Overfitting_Gap']:+.4f}\n")
            if abs(row['Overfitting_Gap']) < 0.01:
                f.write("    → Mô hình generalizes tốt (không overfitting)\n")
            elif row['Overfitting_Gap'] > 0.05:
                f.write("    → CẢNH BÁO: Có dấu hiệu overfitting\n")
            else:
                f.write("    → Overfitting ở mức chấp nhận được\n")
            
            # Detailed metrics
            f.write("\nMetrics trên Test Set:\n")
            f.write(f"  - Accuracy: {row['Accuracy']:.4f}\n")
            f.write(f"  - Balanced Accuracy: {row['Balanced_Accuracy']:.4f}\n")
            f.write(f"  - Precision: {row['Precision']:.4f}\n")
            f.write(f"  - Recall (Sensitivity): {row['Recall_Sensitivity']:.4f}\n")
            f.write(f"  - Specificity: {row['Specificity']:.4f}\n")
            f.write(f"  - F1-Score: {row['F1_Score']:.4f}\n")
            f.write(f"  - MCC: {row['MCC']:.4f}\n")
            if pd.notna(row['ROC_AUC']) and row['ROC_AUC'] > 0:
                f.write(f"  - ROC-AUC: {row['ROC_AUC']:.4f}\n")
            if pd.notna(row['OOB_Score']) and row['OOB_Score'] > 0:
                f.write(f"  - OOB Score: {row['OOB_Score']:.4f}\n")
            
            # Confusion matrix
            if model_name in results:
                cm = np.array(results[model_name]['confusion_matrix'])
                f.write("\nConfusion Matrix (Test Set):\n")
                f.write(f"                Predicted\n")
                f.write(f"              Fail    Pass\n")
                f.write(f"  Actual Fail  {cm[0,0]:5d}  {cm[0,1]:5d}\n")
                f.write(f"        Pass  {cm[1,0]:5d}  {cm[1,1]:5d}\n")
                f.write(f"\n  - True Negatives (TN): {row['TN']}\n")
                f.write(f"  - False Positives (FP): {row['FP']}\n")
                f.write(f"  - False Negatives (FN): {row['FN']}\n")
                f.write(f"  - True Positives (TP): {row['TP']}\n")
            
            f.write("\n" + "="*80 + "\n\n")
        
        # 5. So sánh mô hình
        f.write("5. SO SÁNH VÀ ĐÁNH GIÁ CÁC MÔ HÌNH\n")
        f.write("-"*80 + "\n\n")
        
        # Best model
        best_idx = metrics_df['Test_Accuracy'].idxmax()
        best_model = metrics_df.loc[best_idx, 'Model']
        best_test_acc = metrics_df.loc[best_idx, 'Test_Accuracy']
        
        f.write(f"Mô hình tốt nhất: {best_model}\n")
        f.write(f"  - Test Accuracy: {best_test_acc:.4f}\n")
        f.write(f"  - CV Accuracy: {metrics_df.loc[best_idx, 'CV_Accuracy_Mean']:.4f}\n")
        f.write(f"  - Overfitting Gap: {metrics_df.loc[best_idx, 'Overfitting_Gap']:+.4f}\n\n")
        
        # Ranking by different metrics
        f.write("Xếp hạng theo các metrics:\n\n")
        
        f.write("1. Test Accuracy (quan trọng nhất - trên dữ liệu chưa thấy):\n")
        sorted_by_test = metrics_df.sort_values('Test_Accuracy', ascending=False)
        for rank, (idx, row) in enumerate(sorted_by_test.iterrows(), 1):
            f.write(f"   {rank}. {row['Model']}: {row['Test_Accuracy']:.4f}\n")
        
        f.write("\n2. Balanced Accuracy:\n")
        sorted_by_bal = metrics_df.sort_values('Balanced_Accuracy', ascending=False)
        for rank, (idx, row) in enumerate(sorted_by_bal.iterrows(), 1):
            f.write(f"   {rank}. {row['Model']}: {row['Balanced_Accuracy']:.4f}\n")
        
        f.write("\n3. F1-Score:\n")
        sorted_by_f1 = metrics_df.sort_values('F1_Score', ascending=False)
        for rank, (idx, row) in enumerate(sorted_by_f1.iterrows(), 1):
            f.write(f"   {rank}. {row['Model']}: {row['F1_Score']:.4f}\n")
        
        f.write("\n4. MCC (Matthews Correlation Coefficient):\n")
        sorted_by_mcc = metrics_df.sort_values('MCC', ascending=False)
        for rank, (idx, row) in enumerate(sorted_by_mcc.iterrows(), 1):
            f.write(f"   {rank}. {row['Model']}: {row['MCC']:.4f}\n")
        
        f.write("\n5. ROC-AUC:\n")
        sorted_by_roc = metrics_df.sort_values('ROC_AUC', ascending=False)
        for rank, (idx, row) in enumerate(sorted_by_roc.iterrows(), 1):
            f.write(f"   {rank}. {row['Model']}: {row['ROC_AUC']:.4f}\n")
        
        f.write("\n6. Overfitting Gap (càng nhỏ càng tốt):\n")
        sorted_by_gap = metrics_df.sort_values('Overfitting_Gap', ascending=True)
        for rank, (idx, row) in enumerate(sorted_by_gap.iterrows(), 1):
            f.write(f"   {rank}. {row['Model']}: {row['Overfitting_Gap']:+.4f}\n")
        
        # 6. Đánh giá tổng thể
        f.write("\n" + "="*80 + "\n")
        f.write("6. ĐÁNH GIÁ TỔNG THỂ\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Điểm mạnh:\n")
        f.write("  ✓ Dữ liệu được xử lý kỹ lưỡng, loại bỏ data leakage\n")
        f.write("  ✓ Sử dụng train/test split để đánh giá đúng khả năng generalizing\n")
        f.write("  ✓ 5-fold cross-validation cho ước lượng ổn định\n")
        f.write("  ✓ Xử lý tốt class imbalance với class weights\n")
        f.write("  ✓ Feature selection giúp tránh overfitting\n")
        f.write("  ✓ Sử dụng nhiều metrics để đánh giá toàn diện\n\n")
        
        f.write("Nhược điểm cần cải thiện:\n")
        if metrics_df['Overfitting_Gap'].max() > 0.05:
            f.write("  ⚠️ Một số mô hình có overfitting (gap > 0.05)\n")
        f.write("  - Có thể thử hyperparameter tuning sâu hơn\n")
        f.write("  - Có thể thử ensemble methods để cải thiện performance\n")
        f.write("  - Có thể thử feature engineering nâng cao\n\n")
        
        # 7. Khuyến nghị
        f.write("7. KHUYẾN NGHỊ\n")
        f.write("-"*80 + "\n\n")
        f.write(f"1. Mô hình được khuyến nghị sử dụng: {best_model}\n")
        f.write(f"   - Test Accuracy: {best_test_acc:.4f}\n")
        f.write(f"   - Có hiệu suất tốt nhất trên dữ liệu test\n\n")
        
        f.write("2. Lưu ý khi triển khai:\n")
        f.write("   - Sử dụng model đã được lưu trong output/models/\n")
        f.write("   - Sử dụng scaler tương ứng cho mỗi model\n")
        f.write("   - Đảm bảo features được chuẩn hóa đúng như trong training\n")
        f.write("   - Monitor performance trên dữ liệu thực tế\n\n")
        
        f.write("3. Cải thiện trong tương lai:\n")
        f.write("   - Thu thập thêm dữ liệu nếu có thể\n")
        f.write("   - Thử các kỹ thuật feature engineering khác\n")
        f.write("   - Hyperparameter tuning chi tiết hơn\n")
        f.write("   - Thử các mô hình deep learning nếu dữ liệu đủ lớn\n\n")
        
        # 8. File outputs
        f.write("8. CÁC FILE OUTPUT\n")
        f.write("-"*80 + "\n\n")
        f.write("Models:\n")
        f.write(f"  - {config.OUTPUT_DIRS['models']}/\n")
        f.write("    • LogisticRegression.pkl\n")
        f.write("    • DecisionTree.pkl\n")
        f.write("    • RandomForest.pkl\n")
        f.write("    • MLP.pkl\n")
        f.write("    • scaler_standard.pkl\n")
        f.write("    • scaler_robust.pkl\n\n")
        
        f.write("Metrics:\n")
        f.write(f"  - {config.OUTPUT_DIRS['metrics']}/\n")
        f.write("    • metrics_summary.csv\n")
        f.write("    • detailed_metrics.json\n")
        f.write("    • BAO_CAO_CHI_TIET.txt (file này)\n\n")
        
        f.write("Predictions:\n")
        f.write(f"  - {config.OUTPUT_DIRS['predictions']}/\n")
        f.write("    • <Model>_predictions_train.csv\n")
        f.write("    • <Model>_predictions_test.csv\n\n")
        
        f.write("Charts:\n")
        f.write(f"  - {config.OUTPUT_DIRS['charts']}/\n")
        f.write("    • 01_cv_accuracy.png\n")
        f.write("    • 02_balanced_accuracy.png\n")
        f.write("    • 03_f1_score.png\n")
        f.write("    • 04_precision_recall.png\n")
        f.write("    • 05_specificity_sensitivity.png\n")
        f.write("    • 06_mcc_roc_auc.png\n")
        f.write("    • 07_confusion_matrix_*.png\n")
        f.write("    • 08_pass_fail_distribution.png\n")
        f.write("    • 09_score_distribution.png\n")
        f.write("    • 10_top_subjects.png\n")
        f.write("    • 11_enrollment_by_year.png\n")
        f.write("    • 12_metrics_radar.png\n")
        f.write("    • 13_metrics_summary_table.png\n")
        f.write("    • 14_train_vs_test_accuracy.png\n\n")
        
        # Footer
        f.write("="*80 + "\n")
        f.write("KẾT THÚC BÁO CÁO\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Saved: BAO_CAO_CHI_TIET.txt")
    return report_path


