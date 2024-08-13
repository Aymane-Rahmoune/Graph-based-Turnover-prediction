import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error 


def evaluate_regression(y_true, y_pred, limit=float('inf'), print_scores=False):
    """
    Evaluate regression model performance for y_true values below a certain limit.
    
    Parameters:
    - y_true: array-like of shape (n_samples,) - True values of the target.
    - y_pred: array-like of shape (n_samples,) - Predicted values by the model.
    - limit: float - The upper limit for y_true values to be considered in the evaluation.
    - print_scores: bool, optional (default=False) - Whether to print the scores.
    
    Returns:
    - scores: dict - A dictionary containing MAE, MSE, RMSE, and R² scores for y_true values below the specified limit.
    """
    # Filter y_true and y_pred for y_true values below the limit
    indices_below_limit = y_true < limit
    y_true_filtered = y_true[indices_below_limit]
    y_pred_filtered = y_pred[indices_below_limit]
    
    # Compute the metrics using the filtered arrays
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
    mse = mean_squared_error(y_true_filtered, y_pred_filtered)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_filtered, y_pred_filtered)
    mape = mean_absolute_percentage_error(y_true_filtered, y_pred_filtered)
    medae = median_absolute_error(y_true_filtered, y_pred_filtered)
    
    scores = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE':mape,
        'MedAE':medae,
    }
    
    if print_scores:
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R²): {r2:.3f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.3f}")
        print(f"Median Absolute Error  (MedAE): {medae:.3f}")
    return scores

def aggregate_cv_scores(cv_scores):
    """
    Aggregate cross-validation scores by computing mean, standard deviation, and Variance (%) for each metric.

    Parameters:
    - cv_scores: list of dicts - Each dict contains evaluation metrics for a single fold.

    Returns:
    - aggregated_scores: dict - A dictionary with keys for each metric, each key maps to another dict
      containing the 'mean', 'std' (standard deviation), and 'Variance (%)' of that metric across all folds.
    """
    scores_df = pd.DataFrame(cv_scores)
    aggregated_scores = {}
    for metric in scores_df:
        mean, std = scores_df[metric].mean(), scores_df[metric].std()
        aggregated_scores[metric] = {
            'Mean': round(mean, 2), 
            'Std': round(std, 2), 
            'Var (%)': round((std / mean) * 100, 2) if mean else 0
        }
    return aggregated_scores
    
def update_or_add_evaluation_results(results_df, model_name, scores):
    """
    Update or add evaluation results for a model to the results DataFrame, including mean, standard deviation, and coefficient of variation for scores, with numbers rounded to 2 decimal places.
    """
    new_row = {'Model': model_name}
    
    for metric, values in scores.items():
        new_row[f'{metric} Mean'] = values['Mean']
        new_row[f'{metric} Std'] = values['Std']
        new_row[f'{metric} Var (%)'] = values['Var (%)']

    if model_name in results_df['Model'].values:
        idx = results_df.index[results_df['Model'] == model_name][0]
        for key, value in new_row.items():
            results_df.at[idx, key] = value
    else:
        results_df = pd.concat([results_df, pd.DataFrame([new_row], index=[0])], ignore_index=True)
    
    return results_df

def plot_predictions_vs_true(y_true, y_pred, limit=10**5, title='Predicted vs True Values', save_image=False, image_path='predictions_vs_true.png'):
    """
    This function plots the predicted values against the true values for y_true values below a certain limit,
    along with a line representing the ideal x=y prediction.
    
    Parameters:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    limit (float): The upper limit for y_true values to be considered in the plot
    title (str): The title of the plot
    """
    # Filter y_true and y_pred for y_true values below the limit
    indices_below_limit = y_true < limit
    y_true_filtered = y_true[indices_below_limit]
    y_pred_filtered = y_pred[indices_below_limit]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_filtered, y_pred_filtered, alpha=0.3, label='Predictions')
    plt.plot([min(y_true_filtered), max(y_true_filtered)], [min(y_true_filtered), max(y_true_filtered)], color='red', label='Ideal Prediction (x=y)')
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    if save_image:
        plt.savefig(image_path)
    plt.show()


def plot_distribution_difference(y_true, y_pred, bins=30, title='Distribution of Predicted vs True Values', save_image=False, image_path='distribution_difference.png'):
    """
    Plot the distribution of predicted values and true values to compare their distributions.
    
    Parameters:
    - y_true: array-like, true target values.
    - y_pred: array-like, predicted target values.
    - bins: int, number of bins for the histogram.
    - title: str, title of the plot.
    """
    plt.figure(figsize=(10, 7))
    plt.hist(y_true, bins=bins, alpha=0.5, label='True Values')
    plt.hist(y_pred, bins=bins, alpha=0.5, label='Predicted Values')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_image:
        plt.savefig(image_path)
    plt.show()

def plot_ape_distribution_limited(y_true, y_pred, ape_limit=1000, bins=30, title='Distribution of APE with APE < 1000%', save_image=False, image_path='ape_distribution.png'):
    """
    Plot the distribution of Absolute Percentage Errors (APE) where APE is below a specified limit.
    
    Parameters:
    - y_true: array-like, true target values.
    - y_pred: array-like, predicted target values.
    - ape_limit: float, limit for the absolute percentage errors to be considered.
    - bins: int, number of bins for the histogram.
    - title: str, title of the plot.
    """
    # Calculate APE
    ape = np.abs((y_true - y_pred) / y_true) * 100
    
    # Filter APEs below the limit
    ape_filtered = ape[ape < ape_limit]
    
    plt.figure(figsize=(10, 7))
    plt.hist(ape_filtered, bins=bins, alpha=0.7, color='skyblue')
    plt.xlabel('Absolute Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    if save_image:
        plt.savefig(image_path)
    plt.show()

def plot_ape_scatter_limited(y_true, y_pred, ape_limit=1000, title='Scatter Plot of APE with APE < 1000%', save_image=False, image_path='ape_scatter.png'):
    """
    Create a scatter plot of Absolute Percentage Errors (APE) against true values where APE is below a specified limit.
    
    Parameters:
    - y_true: array-like, true target values.
    - y_pred: array-like, predicted target values.
    - ape_limit: float, limit for the absolute percentage errors to be considered.
    - title: str, title of the plot.
    """
    # Calculate APE
    ape = np.abs((y_true - y_pred) / y_true) * 100

    # Apply limit
    indices_below_limit = ape < ape_limit
    y_true_filtered = y_true[indices_below_limit]
    ape_filtered = ape[indices_below_limit]
    
    plt.figure(figsize=(10, 7))
    plt.scatter(y_true_filtered, ape_filtered, alpha=0.7, color='tomato')
    plt.xlabel('True Values')
    plt.ylabel('Absolute Percentage Error (%)')
    plt.title(title)
    plt.grid(True)
    if save_image:
        plt.savefig(image_path)
    plt.show()