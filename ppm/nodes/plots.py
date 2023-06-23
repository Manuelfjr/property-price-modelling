import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame

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
