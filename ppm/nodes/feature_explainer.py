import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import shap
import pandas as pd

def shap_explainer(model: list,
                   X: pd.DataFrame) -> Figure:
    explainer = shap.Explainer(model[0])
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values,
                    X,
                    plot_type="violin",
                    color_bar=False, show=False)
    plt.colorbar(label='SHAP Value')

def tree_explainer(model: list,
                   features_names: list,
                    n_features: int) -> list:
    if hasattr(model[0], "feature_importances_"):
        fe_values = model[0].feature_importances_
    features_importance = pd.DataFrame(
        fe_values,
        index = features_names,
        columns = ["fe"]
    ).sort_values("fe", ascending = False)
    features_selected = list(
        features_importance
        .head(n_features)
        .index
    )
    features_importance = (
        features_importance
        .reset_index()
        .rename(columns = {"index": "features"})
    )
    return features_importance, features_selected