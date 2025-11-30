import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Pie chart + Bar chart tỷ lệ Fraud/Normal
def Fraud_Normal(normal_count , fraud_count):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.pie([normal_count, fraud_count], 
            labels=[f'Normal\n{normal_count:,}', f'Fraud\n{fraud_count:,}'],
            autopct='%1.3f%%', 
            colors=['#66b3ff', '#ff4444'], 
            explode=(0, 0.1), 
            startangle=90,
            textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax1.set_title('Fraud vs Normal Transaction Rate', fontsize=16, pad=20)

    ax2.bar(['Normal', 'Fraud'], [normal_count, fraud_count], 
        color=['#66b3ff', '#ff4444'], edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of transactions')
    ax2.set_title('Number of classes')
    for i, v in enumerate([normal_count, fraud_count]):
        ax2.text(i, v + 5000, f'{v:,}', ha='center', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

# Phân bố Transac Amount + Time
def Transaction_Time_Amount(amount_val , time_val ):
    fig = plt.figure(figsize=(18, 4))

    # Amount
    plt.subplot(1, 2, 1)
    sns.histplot(amount_val, kde=True, color='r', stat="density")
    plt.title('Distribution of Transaction Amount', fontsize=14)

    # Time
    plt.subplot(1, 2, 2)
    sns.histplot(time_val, kde=True, color='b', stat="density")
    plt.title('Distribution of Transaction Time', fontsize=14)

    plt.tight_layout()
    plt.show()

def Plot_Amount_By_Hour(time_col, amount_col, y):
    """
    Vẽ biểu đồ tổng Amount theo Hour cho Normal và Fraud
    
    Parameters:
    - time_col: mảng thời gian (seconds)
    - amount_col: mảng Amount
    - y: mảng nhãn Class (0=Normal, 1=Fraud)
    """
    # Chuyển seconds sang hours
    hours = (time_col / 3600).astype(int)
    
    # Tách data theo Class
    mask_normal = (y == 0)
    mask_fraud = (y == 1)
    
    hours_normal = hours[mask_normal]
    amount_normal = amount_col[mask_normal]
    
    hours_fraud = hours[mask_fraud]
    amount_fraud = amount_col[mask_fraud]
    
    # Tính tổng Amount cho mỗi Hour (Normal)
    unique_hours_normal = np.unique(hours_normal)
    sum_by_hour_normal = np.array([
        amount_normal[hours_normal == h].sum() 
        for h in unique_hours_normal
    ])
    
    # Tính tổng Amount cho mỗi Hour (Fraud)
    unique_hours_fraud = np.unique(hours_fraud)
    sum_by_hour_fraud = np.array([
        amount_fraud[hours_fraud == h].sum() 
        for h in unique_hours_fraud
    ])
    
    # Vẽ biểu đồ
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
    
    # Normal transactions
    sns.lineplot(ax=ax1, x=unique_hours_normal, y=sum_by_hour_normal)
    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Total Amount", fontsize=12)
    ax1.set_title("Normal Transactions", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Fraud transactions
    sns.lineplot(ax=ax2, x=unique_hours_fraud, y=sum_by_hour_fraud, color="red")
    ax2.set_xlabel("Hour", fontsize=12)
    ax2.set_ylabel("Total Amount", fontsize=12)
    ax2.set_title("Fraud Transactions", fontsize=14, fontweight='bold', color='red')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Total Amount by Hour", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def Plot_Transactions_By_Hour(time_col, y):
    """
    Vẽ biểu đồ số lượng giao dịch theo Hour cho Normal và Fraud
    """
    # Chuyển seconds sang hours
    hours = (time_col / 3600).astype(int)
    
    # Tách data theo Class
    mask_normal = (y == 0)
    mask_fraud = (y == 1)
    
    hours_normal = hours[mask_normal]
    hours_fraud = hours[mask_fraud]
    
    # Đếm số lượng transactions cho mỗi Hour (Normal)
    unique_hours_normal = np.unique(hours_normal)
    count_by_hour_normal = np.array([
        np.sum(hours_normal == h) 
        for h in unique_hours_normal
    ])
    
    # Đếm số lượng transactions cho mỗi Hour (Fraud)
    unique_hours_fraud = np.unique(hours_fraud)
    count_by_hour_fraud = np.array([
        np.sum(hours_fraud == h) 
        for h in unique_hours_fraud
    ])
    
    # Vẽ biểu đồ
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
    
    # Normal transactions
    sns.lineplot(ax=ax1, x=unique_hours_normal, y=count_by_hour_normal)
    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Transactions", fontsize=12)
    ax1.set_title("Normal Transactions", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Fraud transactions
    sns.lineplot(ax=ax2, x=unique_hours_fraud, y=count_by_hour_fraud, color="red")
    ax2.set_xlabel("Hour", fontsize=12)
    ax2.set_ylabel("Transactions", fontsize=12)
    ax2.set_title("Fraud Transactions", fontsize=14, fontweight='bold', color='red')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Total Number of Transactions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def Plot_Average_Amount_By_Hour(time_col, amount_col, y):
    """
    Vẽ biểu đồ trung bình Amount theo Hour cho Normal và Fraud
    """
    # Chuyển seconds sang hours
    hours = (time_col / 3600).astype(int)
    
    # Tách data theo Class
    mask_normal = (y == 0)
    mask_fraud = (y == 1)
    
    hours_normal = hours[mask_normal]
    amount_normal = amount_col[mask_normal]
    
    hours_fraud = hours[mask_fraud]
    amount_fraud = amount_col[mask_fraud]
    
    # Tính trung bình Amount cho mỗi Hour (Normal)
    unique_hours_normal = np.unique(hours_normal)
    mean_by_hour_normal = np.array([
        amount_normal[hours_normal == h].mean() 
        for h in unique_hours_normal
    ])
    
    # Tính trung bình Amount cho mỗi Hour (Fraud)
    unique_hours_fraud = np.unique(hours_fraud)
    mean_by_hour_fraud = np.array([
        amount_fraud[hours_fraud == h].mean() 
        for h in unique_hours_fraud
    ])
    
    # Vẽ biểu đồ
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 6))
    
    # Normal transactions
    sns.lineplot(ax=ax1, x=unique_hours_normal, y=mean_by_hour_normal)
    ax1.set_xlabel("Hour", fontsize=12)
    ax1.set_ylabel("Mean", fontsize=12)
    ax1.set_title("Normal Transactions", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Fraud transactions
    sns.lineplot(ax=ax2, x=unique_hours_fraud, y=mean_by_hour_fraud, color="red")
    ax2.set_xlabel("Hour", fontsize=12)
    ax2.set_ylabel("Mean", fontsize=12)
    ax2.set_title("Fraud Transactions", fontsize=14, fontweight='bold', color='red')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Average Amount of Transactions", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def Plot_Amount_Boxplot(amount_col, y):
    """
    Vẽ boxplot Amount theo Class (Normal vs Fraud)
    - ax1: Có outliers (showfliers=True)
    - ax2: Không có outliers (showfliers=False)
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    
    # Tách data theo Class
    amount_normal = amount_col[y == 0]
    amount_fraud = amount_col[y == 1]
    
    # Chuẩn bị data cho boxplot (dạng list of arrays)
    data_to_plot = [amount_normal, amount_fraud]
    labels = ['0', '1']  # Class labels
    
    # Boxplot với outliers
    bp1 = ax1.boxplot(data_to_plot, 
                      labels=labels,
                      patch_artist=True,
                      showfliers=True)
    
    # Tô màu cho boxplot 1 (giống palette PRGn)
    colors = ['#af8dc3', '#7fbf7b']  # Màu tím và xanh lá (PRGn palette)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_xlabel("Class", fontsize=12)
    ax1.set_ylabel("Amount", fontsize=12)
    ax1.set_title("With Outliers", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Boxplot không có outliers
    bp2 = ax2.boxplot(data_to_plot,
                      labels=labels,
                      patch_artist=True,
                      showfliers=False)
    
    # Tô màu cho boxplot 2
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xlabel("Class", fontsize=12)
    ax2.set_ylabel("Amount", fontsize=12)
    ax2.set_title("Without Outliers", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def Plot_Correlation_Heatmap(X, y, feature_names=None):
    """
    Vẽ biểu đồ Heatmap tương quan Pearson từ numpy array (không dùng Pandas).
    
    Parameters:
    - X: Mảng đặc trưng (features)
    - y: Mảng nhãn (target/class)
    - feature_names: List tên các cột đặc trưng (tùy chọn)
    """
    # 1. Ghép X và y để tính tương quan giữa Features và Class
    # data_combined sẽ có kích thước (n_samples, n_features + 1)
    data_combined = np.column_stack((X, y))
    
    # 2. Tính ma trận tương quan Pearson
    # rowvar=False: báo cho numpy biết mỗi cột là một biến (variable)
    corr_matrix = np.corrcoef(data_combined, rowvar=False)
    
    # 3. Xử lý tên cột (Labels) để hiển thị
    if feature_names is None:
        # Nếu không truyền tên, tự tạo: F0, F1,... Class
        n_features = X.shape[1]
        labels = [f"F{i}" for i in range(n_features)] + ["Class"]
    else:
        # Nếu đã truyền tên các feature, thêm "Class" vào cuối cho khớp với data_combined
        # Copy list để tránh sửa đổi list gốc bên ngoài hàm
        labels = list(feature_names) + ["Class"]

    # 4. Vẽ biểu đồ
    plt.figure(figsize=(14, 14))
    plt.title('Credit Card Transactions features correlation plot (Pearson)', fontsize=16)
    
    sns.heatmap(
        corr_matrix, 
        xticklabels=labels, 
        yticklabels=labels, 
        linewidths=.1, 
        cmap="Reds",
        annot=False,  # Tắt số cụ thể vì ma trận lớn sẽ rất rối
        square=True   # Giữ các ô hình vuông
    )
    
    plt.tight_layout()
    plt.show()


def Plot_Evaluation(model, y_test, y_proba):
    """
    Vẽ 3 biểu đồ đánh giá: Training Loss, ROC Curve, và Precision-Recall Curve.
    Hoạt động với model LogisticRegressionNumPy tự viết.
    
    Parameters:
    - model: Object model đã train (có thuộc tính .train_loss)
    - y_test: Nhãn thực tế của tập test (0 hoặc 1)
    - y_proba: Xác suất dự đoán của lớp 1 (output của predict_proba)
    """
    
    # Tạo khung hình 3 cột
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # ---------------------------------------------------------
    # 1. BIỂU ĐỒ TRAINING LOSS
    # ---------------------------------------------------------
    ax1.plot(model.train_loss, label='Train Loss', linewidth=2)
    # Nếu model có lưu validation loss thì vẽ luôn
    if hasattr(model, 'val_loss') and len(model.val_loss) > 0:
        ax1.plot(model.val_loss, label='Val Loss', linewidth=2, linestyle='--')
        
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ---------------------------------------------------------
    # 2. BIỂU ĐỒ ROC CURVE & AUC (Tính bằng NumPy)
    # ---------------------------------------------------------
    # Sắp xếp xác suất giảm dần
    sorted_idx = np.argsort(-y_proba)
    y_sorted = y_test[sorted_idx]
    
    # Tính TPR và FPR tích lũy
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    
    # Tính AUC (Diện tích dưới đường cong)
    # Lưu ý: np.trapezoid mới có ở NumPy 2.0, các bản cũ dùng np.trapz
    try:
        auc_score = np.trapezoid(tpr, fpr)
    except AttributeError:
        auc_score = np.trapz(tpr, fpr)
        
    ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', linewidth=3, color='darkorange')
    ax2.plot([0,1], [0,1], 'k--', alpha=0.5) # Đường chéo ngẫu nhiên
    
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 3. BIỂU ĐỒ PRECISION-RECALL CURVE
    # ---------------------------------------------------------
    thresholds = np.linspace(0, 1, 100) # Giảm xuống 100 điểm để vẽ nhanh hơn
    precisions = []
    recalls = []
    
    # Tính Precision/Recall tại các ngưỡng khác nhau
    for th in thresholds:
        pred_labels = (y_proba >= th).astype(int)
        
        # Tính thủ công không dùng sklearn
        tp = np.sum((y_test == 1) & (pred_labels == 1))
        fp = np.sum((y_test == 0) & (pred_labels == 1))
        fn = np.sum((y_test == 1) & (pred_labels == 0))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0 # 1.0 nếu không bắt được gì (quy ước đồ thị)
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        
    ax3.plot(recalls, precisions, label='PR Curve', linewidth=3, color='blue')
    
    ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Tiêu đề chung
    plt.suptitle('Credit Card Fraud Detection Evaluation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def Plot_Class_Distribution(y, title='Equally Distributed Classes', colors=['#0101DF', '#DF0101']):
    """
    Vẽ biểu đồ đếm số lượng Class (Không dùng vòng lặp for).
    """
    plt.figure(figsize=(8, 6))
    
    # Vẽ Countplot và lấy đối tượng Axes (ax)
    ax = sns.countplot(x=y, hue=y, palette=colors, legend=False)
    # ----------------------
    
    # Thêm số lượng lên đầu cột (Matplotlib > 3.4.0)
    ax.bar_label(ax.containers[0], fontsize=12, padding=3)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class (0: Normal, 1: Fraud)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.show()


def Plot_Feature_Importance(model, feature_names):
    """
    Vẽ biểu đồ tầm quan trọng của đặc trưng dựa trên trọng số (Weights) của Logistic Regression.
    
    Parameters:
    - model: Object model đã train (có thuộc tính .w)
    - feature_names: List tên các đặc trưng (không bao gồm bias)
    """
    # 1. Lấy trọng số (Bỏ qua w[0] vì là bias/intercept)
    # weights tương ứng với feature_names
    weights = model.w[1:] 
    
    # Kiểm tra độ dài cho an toàn
    if len(weights) != len(feature_names):
        print(f"Lỗi: Số lượng weights ({len(weights)}) khác số lượng tên features ({len(feature_names)})")
        return

    # 2. Sắp xếp theo độ lớn tuyệt đối (Magnitude) giảm dần
    # Vì hệ số âm lớn cũng quan trọng ngang hệ số dương lớn
    indices = np.argsort(np.abs(weights))[::-1]
    
    # Sắp xếp tên và giá trị theo index đã tìm được
    names_sorted = np.array(feature_names)[indices]
    weights_sorted = weights[indices]
    
    # 3. Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance (Logistic Regression Coefficients)', fontsize=14, fontweight='bold')
    
    # Dùng seaborn barplot với dữ liệu mảng numpy
    colors = ['#DF0101' if x < 0 else '#0101DF' for x in weights_sorted]
    
    s = sns.barplot(x=names_sorted, y=weights_sorted, palette=colors)
    
    s.set_xticklabels(names_sorted, rotation=90)
    plt.ylabel("Coefficient Weight (Magnitude)", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    
    # Thêm lưới cho dễ nhìn
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()