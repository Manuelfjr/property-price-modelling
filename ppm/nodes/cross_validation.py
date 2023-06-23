from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error
)
from tqdm import tqdm
import numpy as np

def _print_scores(scores: dict) -> None:
    for i, content in scores.items():
        text_mean = f"{i}: {round(np.mean(content[1]), 5)}"
        text_std = f"{i}_std: {round(np.std(content[1]), 5)}"
        print("-" * max((len(text_mean), len(text_std))))
        print(text_mean)
        print(text_std)
        print("-" * max((len(text_mean), len(text_std))))


def cross_validation(X: DataFrame, 
                     y: Series,
                    model: list,
                    cv_splits: int=5,
                    random_state: int=42,
                    show: bool=True) -> dict:
    stratified_kfold = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = {
        "rmse": [mean_squared_error, []],
        "r2": [r2_score, []],
        "mape": [mean_absolute_percentage_error, []],
        "mse": [mean_squared_error, []],
        "median_ae": [median_absolute_error, []],
        "mae": [mean_absolute_error, []]
    }

    for train_index, test_index in tqdm(stratified_kfold.split(X, y)):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        model[0].fit(X_train, y_train)

        y_pred = model[0].predict(X_test)

        args = [y_test, y_pred]
        for i, content in scores.items():
            if i == "rmse":
                content[1].append(
                    np.sqrt(content[0](*args))
                )
            else:
                content[1].append(
                    content[0](*args)
                )
    if show:
        _print_scores(scores)

    return scores
