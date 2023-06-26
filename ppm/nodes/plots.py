import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
import numpy as np

def scatterplot_yx(args_train: list,
                   args_preds: list,
                   fig_args: dict) -> Figure:
    fig, axes = plt.subplots(**fig_args)

    for content in [[0, args_train, "Train"], [1, args_preds, "Test"]]:
        idx, arg, which = content
        axes[idx].plot(arg[0], arg[1],"*")
        axes[idx].set_title(which)
        axes[idx].set_xlabel("True")
        axes[idx].set_ylabel("Prediction")
        axes[idx].grid()
        axes[idx].grid(True)
    
    return fig

def distribution_plot(data: DataFrame,
                      fig_args: dict) -> Figure:
    fig = plt.figure(**fig_args)
    plt.plot(data['y_true'].values, label = "true")
    plt.plot(data['y_pred'].values, label = "pred")
    plt.grid()
    plt.legend()
    return fig


def plot_outliers_histogram(data_input: DataFrame,
                            target: str,
                            lower_limit: float,
                            upper_limit: float,
                            color_within: str = 'grey',
                            color_outside: str = 'red',
                            linestyle: str = '--') -> Figure:
    outliers = data_input[(data_input[target] < lower_limit) | (data_input[target] > upper_limit)][target]
    non_outliers = data_input[(data_input[target] >= lower_limit) & (data_input[target] <= upper_limit)][target]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].axhline(lower_limit, color=color_outside, linestyle=linestyle)
    axes[0].axhline(upper_limit, color=color_outside, linestyle=linestyle)

    for ax in axes:
        ax.grid()

    axes[0].plot(non_outliers.index, non_outliers, '*', color=color_within)
    axes[0].plot(outliers.index, outliers, '*', color=color_outside, label="Outlier")
    axes[1].hist(data_input[target].values)

    axes[1].set_title(f"Histogram of {target}")
    axes[0].set_title(f"Scatter of {target} (n_outliers = {round(100 * len(outliers) / data_input.shape[0], 2)} %)")
    axes[0].set_ylabel(target)
    axes[0].legend()

    return fig


def plot_feature_importance(
        feature_importance: DataFrame,
        top_k: int = 10
        ) -> Figure:
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.plot(range(len(feature_importance.head(top_k).fe)),
              feature_importance.head(top_k).fe,
              "*", 
              c="black")

    for (i, row), x in zip(feature_importance.head(top_k).iterrows(),
                           range(len(feature_importance.head(top_k).fe))):
        axes.annotate(row['features'], 
                      xy=(x, row['fe']), ha='center', va='bottom')
        axes.grid()
    return fig

def plot_predictions(
        data_values: dict,
        metrics_results: dict,
        target: list,
        figsize: tuple
        ) -> Figure:
    fig, axes = plt.subplots(1, len(data_values.items()), figsize = figsize)

    if len(data_values.items()) == 1:
        axes = [axes]

    for ax, (name, content), metric_content in zip(axes, data_values.items(), metrics_results.values()):
        ax.plot(content["y_true"], content["y_true"], 'r', linewidth=2, linestyle='dashed')
        ax.plot(content["y_true"], content["y_pred"], '*', label="Predicted")
        ax.grid(True)
        ax.set_title(f"{name.title()} | R² = {metric_content['r2']:.5f}")
        ax.set_xlabel(f"True {target[0]}")
        ax.set_ylabel(f"Predicted {target[0]}")
        ax.legend()
        
    plt.tight_layout()

    return fig

def plot_true_vs_pred(
    data_values: dict,
    color_n1: str,
    color_n2: str,
    figsize: tuple,
    **kwargs
) -> Figure:
    fig, axes = plt.subplots(1,
                            len(data_values.items()),
                            figsize = figsize)
    
    if len(data_values.items()) == 1:
        axes = [axes]
    for (name, content), ax in zip(data_values.items(), axes):
        ax.plot(content['y_true'].values,
                label="true",
                color=color_n1,
                **kwargs)
        ax.plot(content['y_pred'].values,
                label="pred",
                color=color_n2,
                **kwargs)
        ax.set_title(f"{name.title()} | ρ = {np.corrcoef(content['y_true'].values, content['y_pred'].values)[0, 1]:.5f}")
        ax.grid(True)
    
    plt.legend()
    
    return fig


def plot_true_vs_pred_multiple(
        data_values: dict,
        metrics_results: dict,
        target: list,
        figsize: tuple=(20, 8),
        **kwargs) -> Figure:
    
    fig, axes = plt.subplots(1, len(data_values.keys()), figsize=figsize)
    if len(data_values.keys()) == 1:
        axes = [axes]

    for (idx, (name, content)), metric_content in zip(enumerate(data_values.items()), metrics_results.values()):
        ax = axes[idx]
        ax.plot(content["y_true"], content["y_true"], 'r',**kwargs)# linewidth=2, linestyle='dashed')
        ax.plot(content["y_true"], content["y_pred"], '*',**kwargs)
        ax.set_title(name.title() + r" | $R^{2}$ = " + "{:.5}".format(metric_content["r2"]))
        ax.set_xlabel(f"True {target[0]}")
        ax.set_ylabel(f"Predict {target[0]}")
        ax.grid(True)
    
    return fig
