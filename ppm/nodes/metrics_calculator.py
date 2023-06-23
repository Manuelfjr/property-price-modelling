# metrics
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    mean_absolute_error,
    median_absolute_error
)
import numpy as np
from pandas import DataFrame

def metrics_calculate(args: list) -> dict:
    metrics = {
            "r2": r2_score(
                *args
            ),
            "mape": mean_absolute_percentage_error(
                *args
            ),
            "rmse": np.sqrt(
                mean_squared_error(
                *args
                )
            ),
            "mse": mean_squared_error(
                *args
            ),
            "mae": mean_absolute_error(
                *args
            ),
            "median_ae": median_absolute_error(
                *args
            ),
            "correlation": np.corrcoef(
                *args
            )[0,1],
            "size": len(args[0])
    }
    return metrics

def show_results(metrics: dict,
                 which: str) -> None:
    for me in [[which,metrics]]:
        print(f'-------- [ {me[0]} ] ----------')
        for metric, result in me[1].items():
            print(f"{metric} : {round(result, 4)}")