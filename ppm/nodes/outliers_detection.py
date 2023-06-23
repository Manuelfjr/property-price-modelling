import numpy as np

def detect_outliers(data: list,
                     p: int) -> tuple:
    q1 = np.percentile(data, p)
    q3 = np.percentile(data, 100 - p)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return lower_limit,upper_limit