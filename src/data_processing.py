import numpy as np

def describe(X):
    return {
        'count': X.shape[0],
        'mean': np.mean(X, axis=0),
        'std': np.std(X, axis=0, ddof=1),
        '25%': np.percentile(X, 25, axis=0),
        '50%': np.percentile(X, 50, axis=0),
        '75%': np.percentile(X, 75, axis=0),
        'max': np.max(X, axis=0),
        'min': np.min(X, axis=0)
    }

def print_describe(stats, feature_names=None):
    """
    In transpose: mỗi feature là 1 dòng
    """
    n_features = len(stats['mean'])
    
    if feature_names is None:
        feature_names = [f"Col_{i}" for i in range(n_features)]
    
    stat_order = ['count', 'mean', 'std', '25%', '50%', '75%', 'max', 'min']
    
    # Header
    print(f"\n{'Feature':<15}", end="")
    for stat_name in stat_order:
        print(f"{stat_name:>15}", end="")
    print()
    print("=" * (15 + 12 * len(stat_order)))
    
    # Mỗi feature 1 dòng
    for i, name in enumerate(feature_names):
        print(f"{name:<15}", end="")
        
        print(f"{stats['count']:>17.0f}", end="")  # count
        
        for stat_name in stat_order[1:]:  # bỏ qua count
            print(f"{stats[stat_name][i]:>12.6f}", end="")
        print()
    
    print()



def robust_scale(col):
    median = np.median(col)
    q75, q25 = np.percentile(col, [75, 25])
    iqr = q75 - q25
    iqr = np.where(iqr == 0, 3.0, iqr)
    return (col - median) / iqr

