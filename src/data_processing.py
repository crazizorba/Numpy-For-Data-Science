import numpy as np

def load_data(path="../data/raw/BankChurners.csv"):
    """
    Load dữ liệu và trả về các mảng đã xử lý sơ bộ (chỉ dùng NumPy)
    Trả về: X_num, X_cat, y, feature_names
    """
    raw = np.genfromtxt(path, delimiter=',', dtype=str, encoding='utf-8', skip_header=1)
    raw = np.char.strip(raw, '"')

    # Column indices based on the CSV header
    client_num_idx = 0
    attrition_idx = 1
    num_idx = [2, 4, 10, 11, 12, 13, 14, 17, 18, 19, 20]  # Adjust for your 18 columns: Customer_Age, Dependent_count, Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio
    cat_idx = [3, 5, 6, 7, 8]

    # Extract arrays
    client_num = raw[:, client_num_idx].astype(float)
    y = (raw[:, attrition_idx] == "Attrited Customer").astype(int)  # Binary label: 1 for Attrited, 0 for Existing
    
    num_raw = raw[:,num_idx]
    missing_num_mask = (num_raw == '') | (num_raw == 'NaN')
    num_raw[missing_num_mask] = '0'

    X_num = num_raw.astype(float)
    X_cat = raw[:, cat_idx]

    # Feature names
    num_names = ['Customer_Age', 'Dependent_count', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    cat_names = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    return X_num, X_cat, y, client_num, num_names, cat_names

def handle_unknown(X_cat, strategy="most_frequent"):
    """
    Replace "Unknown" or empty strings with the most frequent value in the column.
    """
    X_clean = X_cat.copy()
    for i in range(X_cat.shape[1]):
        col = X_cat[:, i]
        unknown_mask = (col == "Unknown") | (col == "")
        if np.sum(unknown_mask) == len(col):
            continue
        values, counts = np.unique(col[~unknown_mask], return_counts=True)
        most_common = values[np.argmax(counts)] if len(values) > 0 else ""
        X_clean[unknown_mask, i] = most_common
    return X_clean

def one_hot_encode(X_cat, cat_names):
    """
    One-hot encode categorical features.
    Returns encoded array and new feature names.
    """
    encoded = []
    new_names = []
    for i in range(X_cat.shape[1]):
        unique = np.unique(X_cat[:, i])
        for val in unique:
            encoded_col = (X_cat[:, i] == val).astype(float)
            encoded.append(encoded_col)
            new_names.append(f"{cat_names[i]}_{val}")
    return np.column_stack(encoded), new_names

def handle_outliers(X_num, method="clip_iqr"):
    """
    Xử lý outlier trong numerical data.
    - method: "clip_iqr" (clip to 1.5 IQR bounds)
    - Ý nghĩa: Giảm ảnh hưởng outlier mà không mất data.
    """
    q1 = np.percentile(X_num, 25, axis=0)
    q3 = np.percentile(X_num, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Clip outliers (vectorized)
    X_num_clipped = np.clip(X_num, lower_bound, upper_bound)
    return X_num_clipped

def standardize(X):
    """
    Z-score standardization.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    return (X - mean) / std

def min_max_scale(X):
    """
    Min-Max scaling to [0,1].
    """
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    return (X - min_val) / range_val

def train_val_split(X, y, val_size=0.2, random_state=42):
    """
    Split into train and val sets.
    """
    np.random.seed(random_state)
    n = len(y)
    indices = np.random.permutation(n)
    val_count = int(n * val_size)
    train_idx = indices[val_count:]
    val_idx = indices[:val_count]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def feature_engineering(X_num):
    """
    Create new features from numerical data.
    Returns extended X_num and new feature names.
    """
    # Example new features
    util_ratio = X_num[:, 6] / (X_num[:, 5] + 1e-8)  # Total_Revolving_Bal / Credit_Limit
    avg_trans_amt = X_num[:, 7] / (X_num[:, 8] + 1e-8)  # Total_Trans_Amt / Total_Trans_Ct
    inactive_flag = (X_num[:, 3] >= 3).astype(float)  # Months_Inactive_12_mon >= 3

    new_features = np.column_stack([util_ratio, avg_trans_amt, inactive_flag])
    new_names = ['util_ratio', 'avg_trans_amt', 'inactive_flag']

    return np.hstack([X_num, new_features]), new_names

