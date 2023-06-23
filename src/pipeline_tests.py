# basics
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# feature_importance
import shap

# viz
import matplotlib.pyplot as plt

# models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    mean_absolute_error,
    median_absolute_error
)

# utils
import os

path_root = os.path.join("data")

path_primary = os.path.join(
    path_root, "03_primary"
)

file_path_input_data = os.path.join(
    path_primary, "data_input.csv"
)
file_path_metrics_features_test = os.path.join(
    path_primary, "features_test_metrics.json"
)
file_path_metrics_features_selected = os.path.join(
    path_primary, "features_selected.json"
)

data_input = pd.read_csv(
    file_path_input_data,
    index_col = 0
)

target = [
    "price"
]
cols_to_drop = [
    "cd_setor",
    "ID"
] + target

step = 400
number_of_features = data_input.shape[1]-1
random_state = 42

metrics_all = {}
#print(np.arange(number_of_features, 1, -step))
for i in tqdm(np.arange(number_of_features, 1, -step)):
    if i < step:
        break
    if os.path.exists(file_path_metrics_features_selected):
        with open(file_path_metrics_features_selected, 'r') as json_file:
            features_selected = json.load(json_file)
    if os.path.exists(file_path_metrics_features_test):
        with open(file_path_metrics_features_test, 'r') as json_file:
            metrics_all = json.load(json_file)
    try:
        X = data_input.drop(cols_to_drop, axis=1)[features_selected["features_selected"]]
    except:
        X = data_input.drop(cols_to_drop, axis=1)

    y = data_input[target[0]]

    # Dividindo os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size = 0.2,
                                                        random_state = random_state)

    # Treinando o modelo Random Forest Regressor
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    # Feature Importance
    features_importance = pd.DataFrame(
        rf_model.feature_importances_,
        index = rf_model.feature_names_in_,
        columns = ["fe"]
    ).sort_values("fe", ascending = False)
    features_selected = list(
        features_importance
        .head(number_of_features)
        .index
    )
    features_selected = {
        "features_selected": features_selected
    }
    
    # Extração dos valores SHAP
    explainer = shap.Explainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    args_train = [
        y_train.values,
        rf_model.predict(X_train)
    ]

    args_preds = [
        y_test.values,#y.values,
        rf_model.predict(X_test)
    ]

    metrics_train = {
        "r2": r2_score(
            *args_train
        ),
        "mape": mean_absolute_percentage_error(
            *args_train
        ),
        "rmse": np.sqrt(
            mean_squared_error(
            *args_train
            )
        ),
        "mse": mean_squared_error(
            *args_train
        ),
        "mae": mean_absolute_error(
            *args_train
        ),
        "median_ae": median_absolute_error(
            *args_train
        ),
        "correlation": np.corrcoef(
            *args_train
        )[0,1],
        "size_train": len(args_train[0])
    }

    metrics_pred = {
        "r2": r2_score(
            *args_preds
        ),
        "mape": mean_absolute_percentage_error(
            *args_preds
        ),
        "rmse": np.sqrt(
            mean_squared_error(
            *args_preds
            )
        ),
        "mse": mean_squared_error(
            *args_preds
        ),
        "mae": mean_absolute_error(
            *args_preds
        ),
        "median_ae": median_absolute_error(
            *args_preds
        ),
        "correlation": np.corrcoef(
            *args_preds
        )[0,1],
        "size_test": len(args_preds[0])
    }

    metrics_both = {
        "train": metrics_train,
        "test": metrics_pred
    }
    metrics_all[f"n{i}"] = metrics_both

    number_of_features -= step
    # Save the dictionary as JSON
    with open(file_path_metrics_features_test, 'w') as json_file:
        json.dump(metrics_all, json_file)
        
    with open(file_path_metrics_features_selected, 'w') as json_file:
        json.dump(features_selected, json_file)