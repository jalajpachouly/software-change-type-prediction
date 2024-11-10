import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_model_specific_comparison(metrics_dict, metrics_to_plot=None):
    """
    Plots a model-specific comparison of evaluation metrics for balanced vs. unbalanced datasets.

    Parameters:
    - metrics_dict (dict): Nested dictionary containing evaluation metrics for each model under both conditions.
    - metrics_to_plot (list): List of metrics to plot. If None, plots all available metrics.
    """
    data = []
    for dataset, models in metrics_dict.items():
        for model, metrics in models.items():
            for metric, score in metrics.items():
                # Only include the specified metrics
                if metrics_to_plot is None or metric in metrics_to_plot:
                    data.append({
                        'Dataset': dataset,
                        'Model': model,
                        'Metric': metric,
                        'Score': score
                    })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Specify the order for consistent plotting
    model_order = ['Logistic Regression', 'Linear SVM', 'Multinomial NB', 'Random Forest', 'BERT']
    dataset_order = ['Unbalanced', 'Balanced']

    # Determine the metrics to plot
    if metrics_to_plot is not None:
        metric_order = metrics_to_plot
    else:
        metric_order = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Set the style
    sns.set(style="whitegrid")

    # Create a figure for box plots
    plt.figure(figsize=(10, 6))  # Adjusted size for fewer metrics
    sns.boxplot(
        data=df,
        x='Metric',
        y='Score',
        hue='Dataset',
        palette='Set2',
        order=metric_order
    )
    plt.title('Evaluation Metrics Comparison Across Datasets')
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig('evaluation_metrics_boxplot.png')
    plt.show()

    # Adjust the size based on the number of metrics
    if len(metric_order) == 2:
        aspect_ratio = 1.0
        figure_width = 8
    else:
        aspect_ratio = 0.7
        figure_width = 12

    # Create a figure for bar plots
    g = sns.catplot(
        data=df,
        kind='bar',
        x='Model',
        y='Score',
        hue='Dataset',
        col='Metric',
        col_order=metric_order,
        hue_order=dataset_order,
        order=model_order,
        palette='Set2',
        height=4,
        aspect=aspect_ratio
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Model-Specific Bar Plot Comparison for Balanced vs. Unbalanced Data')
    g.set_axis_labels("Model", "Score")
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison_barplot.png')
    plt.show()

# Updated evaluation metrics based on your data
metrics_dict = {
    'Unbalanced': {
        'Logistic Regression': {'Accuracy': 0.67, 'Precision': 0.72, 'Recall': 0.67, 'F1-Score': 0.63},
        'Linear SVM': {'Accuracy': 0.80, 'Precision': 0.81, 'Recall': 0.80, 'F1-Score': 0.79},
        'Multinomial NB': {'Accuracy': 0.50, 'Precision': 0.32, 'Recall': 0.50, 'F1-Score': 0.33},
        'Random Forest': {'Accuracy': 0.76, 'Precision': 0.77, 'Recall': 0.76, 'F1-Score': 0.74},
        'BERT': {'Accuracy': 0.85, 'Precision': 0.86, 'Recall': 0.85, 'F1-Score': 0.85}
    },
    'Balanced': {
        'Logistic Regression': {'Accuracy': 0.79, 'Precision': 0.79, 'Recall': 0.79, 'F1-Score': 0.79},
        'Linear SVM': {'Accuracy': 0.79, 'Precision': 0.80, 'Recall': 0.79, 'F1-Score': 0.79},
        'Multinomial NB': {'Accuracy': 0.72, 'Precision': 0.76, 'Recall': 0.72, 'F1-Score': 0.72},
        'Random Forest': {'Accuracy': 0.79, 'Precision': 0.80, 'Recall': 0.79, 'F1-Score': 0.78},
        'BERT': {'Accuracy': 0.88, 'Precision': 0.88, 'Recall': 0.88, 'F1-Score': 0.88}
    }
}

# Example usage: Plotting only Accuracy and Precision
plot_model_specific_comparison(metrics_dict, metrics_to_plot=['Accuracy', 'Precision'])
