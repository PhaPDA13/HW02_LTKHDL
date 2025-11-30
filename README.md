## 1.Tiêu đề và mô tả: 

### Credit Card Fraud Detection - Phát Hiện Gian Lận Thẻ Tín Dụng

Mô tả:
Project xây dựng hệ thống phát hiện gian lận thẻ tín dụng bằng cách sử dụng model `Logistic regression`.

Dataset:
284,807 giao dịch từ khoảng 2 ngày
492 giao dịch gian lận (0.172% - cực kỳ mất cân bằng)
30 features: Time, V1-V28 (PCA), Amount
Target: Class (0 = Normal, 1 = Fraud)

---

## 2. Mục lục
1. Tiêu đề và mô tả
2. Mục lục
3. Giới thiệu
4. Dataset
5. Phương pháp (Method)
6. Cài đặt (Installation)
7. Cách sử dụng (Usage)
8. Kết quả (Results)
9. Cấu trúc Project
10. Challenges & Solutions
11. Future Improvements
12. Contributors
13. License

---

## 3. Giới thiệu
### Mô tả bài toán
Phát hiện giao dịch gian lận thẻ tín dụng là bài toán quan trọng trong ngành tài chính. Dữ liệu thực tế bị mất cân bằng, gây khó khăn cho việc xây dựng mô hình ML.

### Động lực & Ứng dụng
- Phát hiện gian lận theo thời gian thực. 
- Giảm thiệt hại tài chính.
- Ứng dụng trong ngân hàng, ví điện tử, fintech.

### Mục tiêu
- Xử lý và chuẩn hoá dữ liệu.
- Implement Logistic Regression bằng NumPy.  
- Tự viết các metrics: ROC, PR Curve, AUC  
- Tối ưu mô hình bằng early stopping, momentum. 

---

## 4. Dataset

### Nguồn dữ liệu
Bộ dữ liệu Credit Card Fraud Detection (Kaggle)
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

- 284.807 giao dịch  
- 492 giao dịch là Fraud

### Mô tả các feature

| Feature | Mô tả |
|--------|-------|
| Time | Thời gian từ giao dịch đầu tiên |
| V1–V28 | Đặc trưng PCA |
| Amount | Số tiền giao dịch |
| Class | 1 = Fraud, 0 = Normal |

### Đặc điểm dữ liệu

| Đặc Điểm | Chi Tiết |
|---------|---------|
| **Kích thước dataset** | 284,807 giao dịch × 31 cột (30 features + 1 target) |
| **Số lượng giao dịch bình thường** | 283,315 (99.828%) |
| **Số lượng giao dịch gian lận** | 492 (0.172%) |
| **Tỷ lệ mất cân bằng** | 1:575 (Fraud : Normal) |
| **PCA Features (V1-V28)** | Đã được chuẩn hóa (mean=0, std≈1) |
| **Time Feature** | Cần xử lý outliers và scaling |
| **Amount Feature** | Cần xử lý outliers (percentile 99.5%) và robust scaling |
| **Missing Values** | Rất ít, xử lý bằng column mean imputation |
| **Data Type** | Float64 (features), Int32 (target) |

**Lưu ý quan trọng:** Sự mất cân bằng cực kỳ nghiêm trọng này đòi hỏi:
- Undersampling hoặc SMOTE để cân bằng dữ liệu
- Stratified K-Fold để đảm bảo mỗi fold có đủ fraud samples
- Sử dụng AUC-ROC thay vì Accuracy làm metric chính

---

## 5. Phương pháp (Method)

### 5.1 Quy trình xử lý dữ liệu

#### Bước 1: Load dữ liệu
```python
data = np.genfromtxt(path_data, delimiter=',', skip_header=1, dtype=str, encoding='utf-8')
X_raw = data[:, :-1].astype(np.float64)
y = data[:, -1].astype(np.int32)
```
- Đọc file CSV sử dụng `numpy.genfromtxt`
- Tách features (X_raw) và target (y)
- Chuyển đổi kiểu dữ liệu: float64 cho features, int32 cho target

#### Bước 2: Xử lý Missing Values
```python
missing_mask = np.isnan(X_raw)
if missing_mask.any():
    col_means = np.nanmean(X_raw, axis=0)
    inds = np.where(missing_mask)
    X_raw[inds] = np.take(col_means, inds[1])
```
- Phát hiện giá trị NaN trong từng cột
- Tính mean của mỗi cột (bỏ qua NaN)
- Điền NaN bằng giá trị mean tương ứng của cột

#### Bước 3: Xử lý Outliers trong Amount
```python
amount_col_idx = 29
p99_5 = np.percentile(X_raw[:, amount_col_idx], 99.5)
X_raw[:, amount_col_idx] = np.clip(X_raw[:, amount_col_idx], None, p99_5)
```
- Tính percentile 99.5% của cột Amount
- Giới hạn các giá trị lớn hơn percentile về mức percentile
- Loại bỏ outliers cực đoan mà không xóa dữ liệu

#### Bước 4: Robust Scaling cho Time & Amount
```python
scaled_time = robust_scale(time_col)    
scaled_amount = robust_scale(amount_col) 
```
- Sử dụng **median** và **IQR** thay vì mean/std
- Ít bị ảnh hưởng bởi outliers so với z-score normalization
- **Công thức**: $x_{scaled} = \frac{x - median}{Q3 - Q1}$

#### Bước 5: Tái cấu trúc Feature Matrix
```python
X_clean = np.delete(X_raw, [0, 29], axis=1)  
X_final = np.column_stack((scaled_amount, scaled_time, X_clean))
```
- Xóa cột Time (index 0) và Amount (index 29) gốc
- Xếp chồng (stack) theo thứ tự: scaled_amount, scaled_time, V1-V28
- **Thứ tự final features**: [scaled_amount, scaled_time, V1, V2, ..., V28]

#### Bước 6: Lưu Processed Data
```python
header = "scaled_amount,scaled_time," + ",".join([f"V{i}" for i in range(1, 29)]) + ",Class"
output_csv = np.hstack([X_final, y.reshape(-1, 1)])
np.savetxt(path_data_processed, output_csv, delimiter=",", header=header, fmt="%.10f")
```
- Tạo header cho các cột
- Ghép X_final và y thành ma trận đầu ra
- Lưu vào CSV với độ chính xác 10 chữ số thập phân

#### Bước 7: Dọn dẹp bộ nhớ
```python
del data, X_raw, X_clean, scaled_amount, scaled_time
```
- Xóa các biến trung gian không còn dùng
- Giải phóng bộ nhớ sau khi xử lý xong

---

### 5.2 Thuật toán sử dụng

#### 1. Hàm Sigmoid (Activation Function)

Hàm sigmoid chuyển đổi output tuyến tính thành xác suất trong khoảng [0, 1]:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Trong đó:
- $z = w^T x + b$ là output tuyến tính
- $w$ là vector trọng số (weights)
- $x$ là vector features
- $b$ là bias term

**Tính chất:** 
- $\sigma(z) \in (0, 1)$ - phù hợp cho binary classification
- $\sigma(0) = 0.5$ - điểm quyết định
- Đạo hàm: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

#### 2. Binary Cross Entropy Loss (BCE)

Hàm mất mát đo lường sự khác biệt giữa dự đoán và nhãn thực:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

Trong đó:
- $N$ là số lượng mẫu
- $y_i$ là nhãn thực (0 hoặc 1)
- $\hat{y}_i = \sigma(z_i)$ là xác suất dự đoán
- Giá trị L càng nhỏ càng tốt

**Giải thích:**
- Nếu $y=1$: $L = -\log(\hat{y})$ (penalize khi dự đoán thấp)
- Nếu $y=0$: $L = -\log(1-\hat{y})$ (penalize khi dự đoán cao)

#### 3. Gradient Descent - Công Thức Cập Nhật Trọng Số

**Đạo hàm loss theo weights:**

$$\nabla_w L = \frac{1}{N} X^T(\hat{y} - y)$$

**Cập nhật weights (learning rule cơ bản):**

$$w := w - \alpha \nabla_w L$$

Trong đó:
- $\alpha$ là learning rate (tốc độ học)
- $X$ là ma trận features ($N \times d$)
- $\hat{y} - y$ là vector sai số

**Cập nhật bias:**

$$b := b - \alpha \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)$$

#### 4. Momentum Optimization

Momentum cải thiện convergence bằng cách giữ hướng gradient trước đó:

$$v_t = \beta v_{t-1} - \alpha \nabla_w L$$

$$w_t = w_{t-1} + v_t$$

Trong đó:
- $v_t$ là velocity/momentum vector tại epoch $t$
- $\beta$ là momentum coefficient (thường = 0.9)
- $\alpha$ là learning rate

**Ưu điểm:**
- Tránh đưa xuống cục bộ (local minima)
- Hội tụ nhanh hơn
- Giảm oscillation

#### 5. Early Stopping

Dừng training khi loss validation không cải thiện trong $K$ epochs liên tiếp:

$$\text{Stop if } \min(L_{val}^{[t-K..t]}) = L_{val}^{[t-K]}$$

**Tác dụng:** Ngăn chặn overfitting, tiết kiệm thời gian huấn luyện

#### 6. ROC Curve & AUC-ROC

**True Positive Rate (Sensitivity/Recall):**

$$TPR = \frac{TP}{TP + FN}$$

**False Positive Rate:**

$$FPR = \frac{FP}{FP + TN}$$

**ROC Curve:** Plot các điểm $(FPR, TPR)$ ứng với từng threshold từ 0 đến 1

**AUC-ROC (Area Under Curve):**

$$AUC = \int_0^1 TPR(FPR) \, dFPR$$

Được tính bằng tích phân hình thang (trapezoidal rule):

$$AUC \approx \sum_{i=1}^{n-1} \frac{(FPR_{i+1} - FPR_i) \cdot (TPR_{i+1} + TPR_i)}{2}$$

**Giải thích:**
- AUC = 1: Mô hình hoàn hảo
- AUC = 0.5: Mô hình random
- AUC > 0.7: Mô hình tốt

#### 7. Evaluation Metrics

**Confusion Matrix Components:**
- **TP (True Positive):** Dự đoán Fraud, thực tế Fraud
- **TN (True Negative):** Dự đoán Normal, thực tế Normal  
- **FP (False Positive):** Dự đoán Fraud, thực tế Normal
- **FN (False Negative):** Dự đoán Normal, thực tế Fraud

**Accuracy:**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision (PPV - Positive Predictive Value):**

$$\text{Precision} = \frac{TP}{TP + FP}$$
*Trong những dự đoán Fraud, bao nhiêu là chính xác?*

**Recall (Sensitivity/TPR):**

$$\text{Recall} = \frac{TP}{TP + FN}$$
*Trong những Fraud thực tế, bao nhiêu được phát hiện?*

**F1-Score (Harmonic Mean):**

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### 5.3 Implement bằng NumPy

#### Tối ưu Hóa Tính Toán:
- **Vector hoá hoàn toàn** bằng `np.dot()` và `np.einsum()` thay vì loop
- **Clip giá trị** $z \in [-250, 250]$ để tránh overflow/underflow trong $e^{-z}$
- **Tính ROC, AUC** bằng cumulative sums thay vì loop từng threshold
- **Tự implement Early Stopping** dựa trên validation loss

---

## 6. Installation & Setup

### Yêu cầu Hệ Thống
- **Python:** 3.11.5+
- **min_ds-env**

### Tải Dataset
1. Truy cập [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
2. Tải file `creditcard.csv`
3. Đặt vào thư mục `Data/raw/`

### Setup
Khi chạy các notebook, hãy chắc chắn rằng phần **Setup Environment** được chạy đầu tiên:

```python
import sys
import os

project_root = os.path.abspath("..")
sys.path.append(project_root)
```

Điều này sẽ thêm thư mục project vào Python's module search path, cho phép import các module từ `src/` directory.

### Kiểm Tra Cài Đặt
Để kiểm tra rằng tất cả các module được cài đặt đúng, chạy lệnh sau trong Python:

```python
import numpy as np
from src.visualization import *
from src.data_processing import *
from src.models import *

print(" NumPy:", np.__version__)
```

---

## 7. Cách sử dụng (Usage)

### Quy Trình Chạy Toàn Bộ Pipeline

Dự án được chia thành 3 notebook chính. Hãy chạy theo thứ tự:

#### **Notebook 1: Data Exploration (01_data_exploration.ipynb)**

Mục đích: Khám phá và phân tích dữ liệu gốc

```bash
Có thể chọn run từng cell hoặc run all cell
```

**Các bước chính:**
1. Setup Environment - cấu hình sys.path
2. Import Libraries - import NumPy và các hàm custom
3. Load Raw Data - đọc file `Data/raw/creditcard.csv`
4. Phân tích attributes - hiểu cấu trúc dữ liệu
5. Kiểm tra class distribution - thấy rõ sự mất cân bằng
6. Khám phá Time & Amount features
7. Vẽ các biểu đồ: 
   - Amount by Hour
   - Transactions by Hour
   - Boxplot Amount vs Class
   - Correlation Heatmap


---

#### **Notebook 2: Data Preprocessing (02_preprocessing.ipynb)**

Mục đích: Xử lý và chuẩn hoá dữ liệu

```bash
Có thể chọn run từng cell hoặc run all cell
```

**Các bước chính:**
1. Load raw data từ `Data/raw/creditcard.csv`
2. **Handle Missing Values** - xử lý giá trị NaN
   ```python
   missing_mask = np.isnan(X_raw)
   col_means = np.nanmean(X_raw, axis=0)
   X_raw[np.where(missing_mask)] = col_means[indices]
   ```
3. **Handle Outliers** - cắt outliers Amount tại percentile 99.5%
   ```python
   p99_5 = np.percentile(X_raw[:, 29], 99.5)
   X_raw[:, 29] = np.clip(X_raw[:, 29], None, p99_5)
   ```
4. **Robust Scaling** - chuẩn hoá Time & Amount features
   ```python
   scaled_amount = robust_scale(X_raw[:, 29])
   scaled_time = robust_scale(X_raw[:, 0])
   ```
5. **Reconstruct Feature Matrix** - ghép các features đã xử lý
6. **Save Processed Data** - lưu vào `Data/processed/data_processed.csv`

---

#### **Notebook 3: Modeling (03_modeling.ipynb)**

Mục đích: Huấn luyện mô hình và đánh giá kết quả

```bash
Có thể chọn run từng cell hoặc run all cell
```

**Các bước chính:**

1. **Load Processed Data**
   ```python
   data = np.genfromtxt('../Data/processed/data_processed.csv', 
                        delimiter=',', skip_header=1)
   X = data[:, :-1].astype(np.float64)
   y = data[:, -1].astype(np.int32)
   ```

2. **Balance Dataset** - cân bằng dữ liệu bằng undersampling
   ```python
   fraud_idx = np.where(y == 1)[0]
   non_fraud_idx = np.where(y == 0)[0]
   sampled_non_fraud = non_fraud_idx[:len(fraud_idx)]
   balanced_idx = np.concatenate([fraud_idx, sampled_non_fraud])
   ```

3. **Stratified K-Fold Cross-Validation** - 5-fold CV
   ```python
   for fold, (train_idx, val_idx) in enumerate(folds):
       model = LogisticRegressionNumPy(lr=0.05, momentum=0.9, 
                                       max_epochs=5000, patience=80)
       model.fit(X_tr_norm, y_tr, X_val_norm, y_val)
   ```

4. **Train-Test Split** - 80/20 split
   ```python
   test_ratio = 0.2
   X_train_final, X_test_final = split_data(X_bal, test_ratio)
   y_train_final, y_test_final = split_labels(y_bal, test_ratio)
   ```

5. **Normalize Data** - Z-score normalization
   ```python
   mean_train = X_train_final.mean(axis=0)
   std_train = X_train_final.std(axis=0)
   X_train_norm = (X_train_final - mean_train) / std_train
   X_test_norm = (X_test_final - mean_train) / std_train
   ```

6. **Train Final Model**
   ```python
   final_model = LogisticRegressionNumPy(lr=0.05, momentum=0.9, 
                                         max_epochs=5000, patience=80)
   final_model.fit(X_train_norm, y_train_final)
   ```

7. **Generate Predictions**
   ```python
   y_proba = final_model.predict_proba(X_test_norm)[:, 1]
   y_pred = (y_proba >= 0.5).astype(np.int32)
   ```

8. **Calculate Metrics**
   ```python
   TP = np.sum((y_test == 1) & (y_pred == 1))
   TN = np.sum((y_test == 0) & (y_pred == 0))
   FP = np.sum((y_test == 0) & (y_pred == 1))
   FN = np.sum((y_test == 1) & (y_pred == 0))
   
   accuracy = (TP + TN) / len(y_test)
   precision = TP / (TP + FP)
   recall = TP / (TP + FN)
   f1 = 2 * precision * recall / (precision + recall)
   ```

9. **Visualize Results** - vẽ ROC Curve, Confusion Matrix, Training Loss

10. **Clean Up Memory** - xóa biến trung gian

**Output:** 
- Model weights & biases
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Visualizations: ROC Curve, Confusion Matrix, Training Loss

---

### Chạy Từng Cell Trong Notebook

1. Chạy **Setup Environment** cell đầu tiên (quan trọng!)
2. Chạy các cell khác lần lượt từ trên xuống
3. Xem output và biểu đồ khi chạy

```python
import sys
import os
project_root = os.path.abspath("..")
sys.path.append(project_root)
```

---


## 8. Kết quả (Results)

### 8.1 Metrics Đạt Được

Mô hình Logistic Regression được huấn luyện trên dữ liệu cân bằng đạt được các kết quả sau:

#### **Kết Quả Thống kê**

| Metric | Giá Trị |
|--------|---------|
| **AUC-ROC** | 0.97683 |
| **Accuracy** | 0.94388 |
| **Precision** | 0.96117 |
| **Recall** | 0.93396 |
| **F1-Score** | 0.94737 |


### 8.2 Trực quan hóa kết quả

![Trực quan hóa kết quả](image-1.png)

### 8.3 So sánh và phân tích

#### 1. Phân tích Chỉ số (Metrics Summary)
| Chỉ số (Metric) | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.97683** | Tỷ lệ rất cao, gần như phân biệt hoàn hảo giữa Normal và Fraud. |
| **Accuracy** | **0.94388** | Tỷ lệ dự đoán đúng tổng thể rất cao trên tập dữ liệu cân bằng. |
| **Precision** | **0.96117** | **Độ tin cậy cao:** Khi mô hình báo "Gian lận", xác suất đúng là 96%. |
| **Recall** | **0.93396** | **Độ nhạy tốt:** Mô hình phát hiện được 93.4% tổng số các ca gian lận thực tế. |
| **F1-Score** | **0.94737** | Sự cân bằng giữa Precision và Recall. |

#### 2. Phân tích Biểu đồ (Visual Analysis)

* **Training Loss:** Đường Loss giảm nhanh trong 500 epochs đầu và đi ngang ổn định sau epoch 1000. Điều này chứng tỏ thuật toán Gradient Descent với Momentum đã hội tụ tốt, không có dấu hiệu dao động (oscillation) hay Overfitting cục bộ.
* **ROC Curve:** Đường cong ôm sát góc trên bên trái, với diện tích dưới đường cong (AUC) đạt ~0.98. Điều này khẳng định mô hình có khả năng phân tách tốt ngay cả ở các ngưỡng (threshold) khác nhau.
* **Precision-Recall Curve:** Đường biểu đồ giữ mức Precision ~ 1.0 (tuyệt đối) trong một khoảng Recall dài. Sự sụt giảm chỉ xảy ra ở mức Recall cao (> 0.85), cho thấy các ca gian lận "khó" mới làm giảm độ chính xác của mô hình.
---

## 9. Cấu trúc Project

```
project-name/
├── README.md                          # Chứa các thông tin về project
├── requirements.txt                   # Chứa các thư viện yêu cầu
├── data/			
│   ├── raw/    
|   |   └── creditcard.csv 	           # Dữ liệu gốc
│   └── processed/ 
|       └── data_processed.csv 	       # Dữ liệu đã qua xử lý
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Khám phá dữ liệu
│   ├── 02_preprocessing.ipynb         # Tiền sử lý dữ liệu
│   └── 03_modeling.ipynb              # Xây dựng model
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Chứa các module cho `01_data_exploration.ipynb` và `02_preprocessing.ipynb`
│   ├── visualization.py       # Chứa các module để visualization
│   └── models.py              # Chứa model(Logistic regression) và các module được implement bằng Numpy
```
---

## 10. Challenges & Solutions

### Khó khăn
- Dữ liệu siêu mất cân bằng  
- ROC/AUC phải tự tính bằng NumPy  
- Numerical instability khi tính sigmoid  

### Giải pháp
- Chọn ra 1 tập sample có số lượng Fraud = Normal
- Clip giá trị z ∈ [-250, 250]  
- Vector hóa toàn bộ để tăng tốc  

---

## 11. Future Improvements
1. Dùng SMOTE tự code thay undersampling  
2. Thêm L2 Regularization  
3. Triển khai Neural Network 
4. Threshold tuning + cost-sensitive learning  
5. Cross-validation mạnh hơn (10-fold)

---

## 12. Contributors
```
- Thông tin tác giả: 
 - Họ và Tên: Nguyễn Đăng Pha
 - Mssv: 23120315

- Contact:
 - Email: 23120315@student.hcmus.edu.vn
 - Github: PhaPDA13

```


## 13. License
MIT License
Copyright (C) 2025 Nguyễn Đăng Pha

