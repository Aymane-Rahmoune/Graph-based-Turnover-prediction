import os
import yaml
import pandas as pd
import torch 

from src import create_graph_data_pipeline
from src import initialize_model, initialize_criterion, initialize_optimizer, initialize_scheduler
from src import MAPELoss
from src import cross_validate_model, evaluate_and_visualize_model_performance
from src import update_or_add_evaluation_results

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():

    #------ Initiate dataframe where to save results ------#
    columns = [
        'Model',
        'MAE Mean', 'MAE Std', 'MAE Var (%)',
        'MSE Mean', 'MSE Std', 'MSE Var (%)',
        'RMSE Mean', 'RMSE Std', 'RMSE Var (%)',
        'R2 Mean', 'R2 Std', 'R2 Var (%)',
        'MAPE Mean', 'MAPE Std', 'MAPE Var (%)',
        'MedAE Mean', 'MedAE Std', 'MedAE Var (%)'
    ]

    # Initialize the DataFrame with these columns
    results_df = pd.DataFrame(columns=columns)

    #------ Load configs ------#
    training_config_path = 'config/training_config.yaml'
    config = load_config(training_config_path)

    #------ Load and prepare the dataset ------#
    transactions_df = pd.read_csv(config['data_path'], delimiter=';')
    node_attributes_df = pd.read_csv(config['node_attributes_path'])
    data = create_graph_data_pipeline(transactions_df, node_attributes_df, target_column='T_LOCAL_MT_ACTIF_SOCIAL')
    
    num_node_features = data.x.size(1)
    num_edge_features = data.edge_attr.size(1)
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")

    #------ Train and Evaluate models ------#
    for model_name, model_config in config['models'].items():
        if model_config['train']:

            # Initialize model
            model = initialize_model(model_name, model_config, num_node_features, num_edge_features)
            
            # Initialize optimizer, criterion, and scheduler
            optimizer = initialize_optimizer(model, model_config.get('optimizer'))
            criterion = initialize_criterion(model_config.get('criterion'))
            scheduler = initialize_scheduler(optimizer, model_config.get('scheduler'))

            #------ Cross Validation ------#
            print("Performing cross-validation...")
            cv_scores = cross_validate_model(
                data,
                model, 
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                val_size=model_config['val_size'],
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                epochs=model_config["epochs"],
                verbose=model_config["verbose"],
                print_every=model_config["print_every"],
                random_state=1,
                )
            results_df = update_or_add_evaluation_results(results_df, model_name, cv_scores)
            print(f"Cross-validation scores:\n {cv_scores} \n")
             
            #------ Plot ------#
            print("Evaluating model and generating plots...")
            scores = evaluate_and_visualize_model_performance(
                data,
                model, 
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                train_mask=None, 
                val_mask=None, 
                val_size=model_config['val_size'],
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                epochs=model_config["epochs"],
                verbose=model_config["verbose"],
                print_every=model_config['print_every'],
                save_plots=model_config['visualization']['save_plots'],
                save_path=model_config['visualization']['save_path'],
                title_prefix=f"{model_name} - ",
                plot_limit=model_config['visualization']['plot_limit'], 
                plot_bins=model_config['visualization']['plot_bins'],
                random_state=1, 
            )
            print("Model evaluation and visualization completed.\n")

    #------ Save the results ------#
    save_path = config.get('save_path_csv_scores', 'output/model_evaluation_results.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    
    print(f"Model evaluation results saved to {save_path}")

if __name__ == '__main__':
    main()


    