# Student Performance Prediction Project

Dự án Machine Learning dự đoán kết quả học tập của sinh viên dựa trên dữ liệu enrollment.

## Cấu trúc dự án

```
.
├── dulieu/                  # Thư mục chứa dữ liệu Excel
├── output/                  # Thư mục chứa kết quả
│   ├── models/             # Các mô hình đã huấn luyện
│   ├── predictions/        # Kết quả dự đoán
│   ├── metrics/            # Báo cáo metrics
│   ├── charts/             # Biểu đồ và hình ảnh
│   └── data/               # Dữ liệu đã xử lý
├── config.py               # File cấu hình
├── data_loader.py          # Module đọc và xử lý dữ liệu
├── model_trainer.py        # Module huấn luyện mô hình
├── visualization.py        # Module tạo biểu đồ
├── main.py                 # Script chính
├── requirements.txt        # Dependencies
└── README.md              # File này
```

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

Lưu ý: Nếu không cài được XGBoost, project vẫn sẽ chạy với 4 mô hình còn lại (LR, DT, RF, MLP).

## Sử dụng

Chạy pipeline chính:
```bash
python main.py
```

Pipeline sẽ thực hiện các bước sau:
1. Đọc tất cả file Excel từ thư mục `dulieu/`
2. Chuẩn hóa header theo mapping alias
3. Merge về bảng enrollment
4. Tạo cột pass_fail (target variable)
5. Huấn luyện 5 mô hình với 5-fold cross-validation:
   - Logistic Regression (LR)
   - Decision Tree (DT)
   - Random Forest (RF)
   - XGBoost
   - MLP (Multi-layer Perceptron)
6. Xuất kết quả vào thư mục `output/`
7. Sinh file `metrics_summary.csv` và các biểu đồ

## Mô hình

- **Logistic Regression**: Mô hình tuyến tính cơ bản
- **Decision Tree**: Cây quyết định
- **Random Forest**: Ensemble của nhiều cây quyết định
- **XGBoost**: Gradient boosting mạnh mẽ
- **MLP**: Neural network với 2 hidden layers

## Output

Sau khi chạy, bạn sẽ có:
- `output/models/`: Các file .pkl chứa mô hình đã huấn luyện
- `output/metrics/metrics_summary.csv`: Bảng tóm tắt metrics
- `output/metrics/detailed_metrics.json`: Metrics chi tiết
- `output/charts/`: Các biểu đồ so sánh mô hình
- `output/data/enrollment_processed.csv`: Dữ liệu đã xử lý

## Metrics được đánh giá

- Cross-Validation Accuracy (Mean & Std)
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix (TP, TN, FP, FN)

# AI_GD_DHBD
