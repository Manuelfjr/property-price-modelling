import matplotlib.pyplot as plt
import shap
from pandas.core.frame import DataFrame

def shap_explainer(model: list,
                   X: DataFrame) -> tuple:
    fig = plt.figure()
    explainer = shap.Explainer(model[0])
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values,
                    X,
                    plot_type="violin",
                    color_bar=False, show=False)
    plt.colorbar(label='SHAP Value')
    return fig, explainer


def tree_explainer(model: list,
                   features_names: list,
                    n_features: int) -> list:
    if hasattr(model[0], "feature_importances_"):
        fe_values = model[0].feature_importances_
    features_importance = DataFrame(
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