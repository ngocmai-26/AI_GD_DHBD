"""
Model training module with 5-fold cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# Try to import XGBoost, make it optional
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available or failed to load runtime (libomp). It will be skipped.")
    print("Fix: brew install libomp (macOS) or pip install xgboost; continuing without XGBoost...")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, balanced_accuracy_score
)
import joblib
import os
import json
from pathlib import Path
import config

# Initialize output directories
for dir_path in config.OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

#chuẩn bị ma trận đặc trưng và nhãn đặc trưng
def prepare_features(df, feature_cols):
    """Prepare features and target for modeling with feature selection"""
    # Select numeric features only
    numeric_features = []
    for col in feature_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype in ['int64', 'float64']:
                    numeric_features.append(col)
            except:
                pass
    
    # Select and convert to float
    X_df = df[numeric_features].copy().astype(float)
    
    # Get target
    y = df[config.TARGET_COLUMN].values
    
    # Remove rows with NaN target
    valid_idx = ~pd.isna(y)
    X_df = X_df.iloc[valid_idx].copy()
    y = y[valid_idx]
    
    # Remove features with zero variance (constant features)
    variance_selector = VarianceThreshold(threshold=0.0)
    X_var = variance_selector.fit_transform(X_df)
    selected_features_mask = variance_selector.get_support()
    numeric_features = [nf for nf, selected in zip(numeric_features, selected_features_mask) if selected]
    X_df = pd.DataFrame(X_var, columns=numeric_features)
    
    print(f"  - After removing zero variance: {len(numeric_features)} features")
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)
    # Replace any residual NaN/Inf with 0
    X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Feature selection: Select top K best features based on f_classif
    # Use up to 50 best features to avoid overfitting (reduced from 100)
    n_features = min(50, X_imputed.shape[1])
    if n_features < X_imputed.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X_imputed, y)
        selected_features_mask = selector.get_support()
        numeric_features = [nf for nf, selected in zip(numeric_features, selected_features_mask) if selected]
        X_imputed = X_selected
        print(f"  - After feature selection: {len(numeric_features)} top features selected")
    
    print(f"\nFeature preparation:")
    print(f"  - Final features: {len(numeric_features)}")
    print(f"  - Samples: {len(X_imputed)}")
    print(f"  - Target distribution: Pass={np.sum(y)}, Fail={np.sum(1-y)}")
    
    return X_imputed, y, numeric_features

#Huấn luyện mô hình
def train_models(X, y, cv_folds=5, test_size=0.2):
    """Train optimized models with 5-fold cross-validation and separate test set"""
    
    #Chia train/test

    
    # Split data into train and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train_full)} samples ({len(X_train_full)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Training target distribution: Pass={np.sum(y_train_full)}, Fail={np.sum(1-y_train_full)}")
    print(f"  Test target distribution: Pass={np.sum(y_test)}, Fail={np.sum(1-y_test)}")

    #Tính class weights (xử lý mất cân bằng)
    # Calculate class weights for imbalanced data (based on training set only)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_full)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_full)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"\nClass weights (to handle imbalance): {class_weight_dict}")

    #Khởi tạo các mô hình
    # Initialize optimized models with better hyperparameters
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=42,          # cố định seed để kết quả lặp lại
            max_iter=10000,           # số vòng lặp tối đa để hội tụ
            class_weight='balanced',  # tự cân bằng lớp Pass/Fail
            C=0.01,                   # hệ số regularization mạnh (giảm overfitting)
            solver='lbfgs',           # thuật toán tối ưu
            penalty='l2',             # chuẩn L2 (ridge) chống overfitting
            n_jobs=-1,                # dùng toàn bộ CPU
            tol=1e-5,                 # ngưỡng dừng hội tụ
            warm_start=False          # không dùng trạng thái training trước đó
        ),
    
        'DecisionTree': DecisionTreeClassifier(
            random_state=42,          # cố định seed
            class_weight='balanced',  # cân bằng lớp
            max_depth=12,             # giới hạn độ sâu để tránh overfit
            min_samples_split=50,     # số mẫu tối thiểu để tách node
            min_samples_leaf=25,      # số mẫu tối thiểu trong leaf
            max_features='sqrt',      # số đặc trưng được chọn tại mỗi split
            min_impurity_decrease=0.001  # yêu cầu giảm impurity tối thiểu
        ),
    
        'RandomForest': RandomForestClassifier(
            n_estimators=300,         # số lượng cây (300 cây)
            random_state=42,          # cố định seed
            class_weight='balanced',  # cân bằng lớp
            max_depth=15,             # giới hạn độ sâu của mỗi cây
            min_samples_split=20,     # số mẫu tối thiểu để split
            min_samples_leaf=10,      # số mẫu tối thiểu trong leaf
            max_features='sqrt',      # số đặc trưng chọn tại mỗi split
            min_impurity_decrease=0.001, # mức giảm impurity tối thiểu
            n_jobs=-1,                # dùng toàn bộ CPU
            oob_score=True,           # tính lỗi OOB (đánh giá ngoài bag)
            bootstrap=True            # bật bootstrap sampling (bagging)
        ),
    
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),  # mạng 2 lớp ẩn: 100 neuron & 50 neuron
            random_state=42,               # cố định seed
            max_iter=2000,                 # số epoch tối đa
            alpha=0.05,                    # hệ số regularization L2
            learning_rate='adaptive',      # giảm learning rate theo hiệu suất
            learning_rate_init=0.001,      # learning rate ban đầu
            early_stopping=True,           # dừng sớm nếu không cải thiện
            validation_fraction=0.1,       # 10% dữ liệu train để validation
            n_iter_no_change=30,           # số epoch không cải thiện để dừng
            batch_size='auto',             # batch size mặc định
            tol=1e-6                       # ngưỡng hội tụ
        )
    }

    # Add XGBoost if available with optimized parameters
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            scale_pos_weight=class_weights[1]/class_weights[0],  # Handle imbalance
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3
        )

    #Chuẩn hóa dữ liệu
    # Use different scalers for different model types
    # Fit scalers on training data ONLY, then transform both train and test
    scaler_standard = StandardScaler()
    X_train_scaled_standard = scaler_standard.fit_transform(X_train_full)
    X_train_scaled_standard = np.clip(X_train_scaled_standard, -5, 5)
    X_test_scaled_standard = scaler_standard.transform(X_test)
    X_test_scaled_standard = np.clip(X_test_scaled_standard, -5, 5)
    
    scaler_robust = RobustScaler()
    X_train_scaled_robust = scaler_robust.fit_transform(X_train_full)
    X_train_scaled_robust = np.clip(X_train_scaled_robust, -5, 5)
    X_test_scaled_robust = scaler_robust.transform(X_test)
    X_test_scaled_robust = np.clip(X_test_scaled_robust, -5, 5)
    
    # Store scalers
    scaler_standard_path = os.path.join(config.OUTPUT_DIRS['models'], 'scaler_standard.pkl')
    scaler_robust_path = os.path.join(config.OUTPUT_DIRS['models'], 'scaler_robust.pkl')
    joblib.dump(scaler_standard, scaler_standard_path)
    joblib.dump(scaler_robust, scaler_robust_path)
    
    # Store test set for final evaluation
    test_set_path = os.path.join(config.OUTPUT_DIRS['data'], 'test_set.pkl')
    joblib.dump({'X_test': X_test, 'y_test': y_test}, test_set_path)
    

    #Thực hiện 5 lần phân tầng
    # Cross-validation on training set only
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    results = {}
    trained_models = {}
    
    print(f"\n{'='*60}")
    print(f"Training models with {cv_folds}-fold cross-validation")
    print(f"Note: CV on training set, final evaluation on test set")
    print(f"{'='*60}")

    #Huấn luyện trên toàn bộ tập train
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Use appropriate scaled features for different models (training set)
        if model_name == 'LogisticRegression':
            X_train = X_train_scaled_standard
            X_test_final = X_test_scaled_standard
        elif model_name == 'MLP':
            X_train = X_train_scaled_robust
            X_test_final = X_test_scaled_robust
        else:
            X_train = X_train_full if isinstance(X_train_full, np.ndarray) else X_train_full
            X_test_final = X_test if isinstance(X_test, np.ndarray) else X_test
        
        # Cross-validation scores on TRAINING SET ONLY
        cv_scoring = 'accuracy'
        cv_scores = cross_val_score(model, X_train, y_train_full, cv=cv, scoring=cv_scoring, n_jobs=-1)
        
        # Also get F1 score from CV
        cv_f1_scores = cross_val_score(model, X_train, y_train_full, cv=cv, scoring='f1', n_jobs=-1)
        
        # Train on full training set
        try:
            model.fit(X_train, y_train_full)
        except Exception as e:
            print(f"  Warning: Model fitting failed: {e}")
            # Try with more conservative parameters
            if model_name == 'LogisticRegression':
                model = LogisticRegression(
                    random_state=42, 
                    max_iter=10000, 
                    class_weight='balanced',
                    C=0.01,
                    solver='saga',
                    penalty='l2'
                )
                model.fit(X_train, y)
            elif model_name == 'MLP':
                model = MLPClassifier(
                    random_state=42, 
                    max_iter=2000,
                    hidden_layer_sizes=(50,),
                    alpha=0.1,
                    learning_rate='constant',
                    learning_rate_init=0.0001
                )
                model.fit(X_train, y_train_full)
            else:
                raise e
        
        # Predictions on TRAINING SET
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Predictions on TEST SET (unseen data)
        y_test_pred = model.predict(X_test_final)
        y_test_pred_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics on TRAINING SET
        train_accuracy = accuracy_score(y_train_full, y_train_pred)
        train_precision = precision_score(y_train_full, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train_full, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train_full, y_train_pred, zero_division=0)
        train_balanced_acc = balanced_accuracy_score(y_train_full, y_train_pred)
        
        # Calculate metrics on TEST SET (unseen data)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
        
        # Use TEST SET metrics as primary (more reliable)
        accuracy = test_accuracy
        precision = test_precision
        recall = test_recall
        f1 = test_f1
        balanced_acc = test_balanced_acc
        
        #Đánh giá trên tập train và tập test
        # Confusion matrix on TEST SET
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Extract TP, TN, FP, FN from confusion matrix
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        elif cm.shape == (1, 1):
            # Only one class predicted
            if len(np.unique(y)) == 1:
                TN, FP, FN, TP = (cm[0,0], 0, 0, 0) if y[0] == 0 else (0, 0, 0, cm[0,0])
            else:
                TN, FP, FN, TP = 0, 0, 0, 0
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
        
        # Calculate additional metrics
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        sensitivity = recall  # Same as recall/TPR
        mcc = matthews_corrcoef(y_test, y_test_pred) if len(np.unique(y_test)) > 1 else 0.0
        
        metrics = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,  # Same as recall
            'specificity': specificity,
            'f1_score': f1,
            'matthews_corrcoef': mcc,
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
        }
        
        # Add OOB score for RandomForest if available
        if model_name == 'RandomForest' and hasattr(model, 'oob_score_'):
            metrics['oob_score'] = model.oob_score_
        
        # ROC AUC on TEST SET
        if y_test_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_test_pred_proba)
            except:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
        
        # Add training metrics for comparison
        metrics['train_accuracy'] = train_accuracy
        metrics['train_f1'] = train_f1
        metrics['test_accuracy'] = test_accuracy
        metrics['test_f1'] = test_f1
        metrics['overfitting_gap'] = train_accuracy - test_accuracy  # Positive = overfitting
        
        # Confusion matrix
        metrics['confusion_matrix'] = cm.tolist()
        
        results[model_name] = metrics
        trained_models[model_name] = model
       
        print(f"  CV Accuracy (train): {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
        print(f"  CV F1-Score (train): {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
        print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f} ← PRIMARY METRIC")
        print(f"  Overfitting Gap: {metrics['overfitting_gap']:+.4f} (train - test)")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  MCC: {metrics['matthews_corrcoef']:.4f}")
        if metrics['roc_auc'] > 0:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        if 'oob_score' in metrics:
            print(f"  OOB Score: {metrics['oob_score']:.4f}")
    
    # Return also test sets for saving predictions
    return results, trained_models, X_train_full, y_train_full, X_test, y_test

def save_models(trained_models):
    """Save trained models"""
    print(f"\nSaving models...")
    for model_name, model in trained_models.items():
        model_path = os.path.join(config.OUTPUT_DIRS['models'], f'{model_name}.pkl')
        joblib.dump(model, model_path)
        print(f"  - Saved {model_name} to {model_path}")

def generate_metrics_summary(results):
    """Generate metrics summary CSV"""
    summary_data = []
    
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'CV_Accuracy_Mean': round(metrics['cv_accuracy_mean'], 4),
            'CV_Accuracy_Std': round(metrics['cv_accuracy_std'], 4),
            'CV_F1_Mean': round(metrics['cv_f1_mean'], 4),
            'CV_F1_Std': round(metrics['cv_f1_std'], 4),
            'Train_Accuracy': round(metrics['train_accuracy'], 4),
            'Test_Accuracy': round(metrics['test_accuracy'], 4),
            'Overfitting_Gap': round(metrics['overfitting_gap'], 4),
            'Accuracy': round(metrics['accuracy'], 4),  # Same as test_accuracy
            'Balanced_Accuracy': round(metrics['balanced_accuracy'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall_Sensitivity': round(metrics['recall'], 4),
            'Specificity': round(metrics['specificity'], 4),
            'F1_Score': round(metrics['f1_score'], 4),
            'MCC': round(metrics['matthews_corrcoef'], 4),
            'ROC_AUC': round(metrics.get('roc_auc', 0.0), 4),
            'OOB_Score': round(metrics.get('oob_score', 0.0), 4) if 'oob_score' in metrics and metrics.get('oob_score') is not None else None,
            'TP': metrics['TP'],
            'TN': metrics['TN'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(config.OUTPUT_DIRS['metrics'], 'metrics_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved metrics summary to {summary_path}")
    
    return df_summary

def save_detailed_metrics(results):
    """Save detailed metrics as JSON"""
    metrics_path = os.path.join(config.OUTPUT_DIRS['metrics'], 'detailed_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed metrics to {metrics_path}")

def save_predictions(trained_models, X_train, y_train, X_test, y_test, feature_names=None):
    """Save predictions from all models on both train and test sets"""
    print(f"\nGenerating predictions...")
    
    # Load scalers
    scaler_standard_path = os.path.join(config.OUTPUT_DIRS['models'], 'scaler_standard.pkl')
    scaler_robust_path = os.path.join(config.OUTPUT_DIRS['models'], 'scaler_robust.pkl')
    scaler_standard = joblib.load(scaler_standard_path)
    scaler_robust = joblib.load(scaler_robust_path)
    
    # Ensure X is numpy array
    X_train_values = X_train if isinstance(X_train, np.ndarray) else X_train
    X_test_values = X_test if isinstance(X_test, np.ndarray) else X_test
    
    X_train_scaled_standard = np.clip(scaler_standard.transform(X_train_values), -5, 5)
    X_train_scaled_robust = np.clip(scaler_robust.transform(X_train_values), -5, 5)
    X_test_scaled_standard = np.clip(scaler_standard.transform(X_test_values), -5, 5)
    X_test_scaled_robust = np.clip(scaler_robust.transform(X_test_values), -5, 5)
    
    for model_name, model in trained_models.items():
        # Use appropriate scaled features
        if model_name == 'LogisticRegression':
            X_train_input = X_train_scaled_standard
            X_test_input = X_test_scaled_standard
        elif model_name == 'MLP':
            X_train_input = X_train_scaled_robust
            X_test_input = X_test_scaled_robust
        else:
            X_train_input = X_train_values
            X_test_input = X_test_values
        
        # Generate predictions on train and test sets
        y_train_pred = model.predict(X_train_input)
        y_train_pred_proba = model.predict_proba(X_train_input)[:, 1] if hasattr(model, 'predict_proba') else None
        
        y_test_pred = model.predict(X_test_input)
        y_test_pred_proba = model.predict_proba(X_test_input)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Save training predictions
        pred_train_df = pd.DataFrame({
            'actual': y_train,
            'predicted': y_train_pred,
        })
        if y_train_pred_proba is not None:
            pred_train_df['probability'] = y_train_pred_proba
        pred_path_train = os.path.join(config.OUTPUT_DIRS['predictions'], f'{model_name}_predictions_train.csv')
        pred_train_df.to_csv(pred_path_train, index=False)
        print(f"  - Saved {model_name} train predictions to {pred_path_train}")
        
        # Save test predictions
        pred_test_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_test_pred,
        })
        if y_test_pred_proba is not None:
            pred_test_df['probability'] = y_test_pred_proba
        pred_path_test = os.path.join(config.OUTPUT_DIRS['predictions'], f'{model_name}_predictions_test.csv')
        pred_test_df.to_csv(pred_path_test, index=False)
        print(f"  - Saved {model_name} test predictions to {pred_path_test}")
    
    print(f"Predictions saved to {config.OUTPUT_DIRS['predictions']}/")

