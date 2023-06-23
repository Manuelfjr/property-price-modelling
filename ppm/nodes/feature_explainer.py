import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import shap
import pandas as pd

def shap_explainer(model: list,
                   X: pd.DataFrame) -> Figure:
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values,
                    X,
                    plot_type="violin",
                    color_bar=False, show=False)
    plt.colorbar(label='SHAP Value')

def tree_explainer(model: list,
                    n_features: int) -> list:
    features_importance = pd.DataFrame(
        model[0].feature_importances_,
        index = model[0].feature_names_in_,
        columns = ["fe"]
    ).sort_values("fe", ascending = False)
    features_selected = list(
        features_importance
        .head(n_features)
        .index
    )
    return features_importance, features_selected