import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from .evaluation_utils import evaluate_regression, aggregate_cv_scores
from .evaluation_utils import plot_predictions_vs_true, plot_distribution_difference, plot_ape_distribution_limited, plot_ape_scatter_limited
from .preprocessing import DataPreprocessor


class ModelTrainer:
    def __init__(self, model, optimizer, criterion, scheduler, data, device, print_every=100, verbose=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.scheduler = scheduler
        self.data = data.to(device)
        self.device = device
        self.best_val_loss = float('inf')
        self.best_model = None
        self.print_every = print_every 
        
        # Setup logging
        self.setup_logging(model.name)

    def setup_logging(self, name_model):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        root_dir = './experiments'
        log_subdir = f'logs/{name_model}_{current_time}'
        model_subdir = f'models/{name_model}_{current_time}'
        log_dir = os.path.join(root_dir, log_subdir)
        model_dir = os.path.join(root_dir, model_subdir)
        self.best_model_path = os.path.join(model_dir, 'best_model.pth')

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)

    def _train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data)
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item(), out[self.data.train_mask], self.data.y[self.data.train_mask]

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
            loss = self.criterion(out[self.data.val_mask], self.data.y[self.data.val_mask])
            return loss.item(), out[self.data.val_mask], self.data.y[self.data.val_mask]

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, y_train_pred, y_train = self._train_epoch()
            val_loss, y_val_pred, y_val = self.evaluate()

            # Calculate additional metrics
            scores_train = evaluate_regression(
                y_train.detach().cpu().numpy(), 
                y_train_pred.detach().cpu().numpy()
            )
            scores_val = evaluate_regression(
                y_val.detach().cpu().numpy(), 
                y_val_pred.detach().cpu().numpy()
            )

            # Step the scheduler on validation loss
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model, self.best_model_path)

            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('MAE/Train', scores_train['MAE'], epoch)
            self.writer.add_scalar('RMSE/Train', scores_train['RMSE'], epoch)
            self.writer.add_scalar('R2/Train', scores_train['R2'], epoch)
            self.writer.add_scalar('MAPE/Train', scores_train['MAPE'], epoch)
            self.writer.add_scalar('MAE/Val', scores_val['MAE'], epoch)
            self.writer.add_scalar('RMSE/Val', scores_val['RMSE'], epoch)
            self.writer.add_scalar('R2/Val', scores_val['R2'], epoch)
            self.writer.add_scalar('MAPE/Val', scores_val['MAPE'], epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
    
            if epoch % self.print_every == 0:
                print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        print(f'Best val loss: {self.best_val_loss}')
        self.writer.close()


def split_train_val_masks(data, val_size=0.2, random_state=1):
    indices = np.arange(data.num_nodes)  # Assuming 'data' has a 'num_nodes' attribute
    train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=random_state)
    
    # Initialize masks to False
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    # Set True for the indices that belong to the respective splits
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    
    return train_mask, val_mask


def train_evaluate_model(
    data,
    model, 
    device,
    train_mask=None, 
    val_mask=None, 
    val_size=0.2,
    optimizer=None,
    criterion=None,
    scheduler=None,
    epochs=500,
    random_state=1, 
    verbose=True,
    print_every=200,
    ):

    #------ Prepare data ------#
    data_copy = copy.deepcopy(data)

    if train_mask is None or val_mask is None:
        print("Generating train and val masks...")
        train_mask, val_mask = split_train_val_masks(data_copy, val_size=val_size, random_state=random_state)
        
    data_copy.train_mask = train_mask
    data_copy.val_mask = val_mask

    preprocessor = DataPreprocessor(data_copy)
    preprocessor.scale_features()
    preprocessor.scale_edge_attributes()

    data_copy.y = torch.sqrt(data_copy.y)

    #------ Train model ------#
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if scheduler is None:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=verbose)

    trainer = ModelTrainer(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        scheduler=scheduler, 
        data=data_copy, 
        device=device, 
        print_every=print_every,
        verbose=verbose,
    )

    trainer.train(epochs)

    #------ Evaluate model ------#
    model.load_state_dict(torch.load(trainer.best_model_path))

    model.eval()
    with torch.no_grad():
        predictions = model(data_copy)

    y_val_pred_transformed = predictions[data_copy.val_mask]
    y_val_transformed = data_copy.y[data_copy.val_mask]

    y_val_pred = y_val_pred_transformed**2
    y_val_pred = y_val_pred.detach().cpu().numpy()

    y_val = y_val_transformed**2
    y_val = y_val.detach().cpu().numpy()

    scores = evaluate_regression(y_val, y_val_pred, print_scores=False)

    return scores, y_val, y_val_pred


def split_data_k_fold(data, n_splits=5, shuffle=True, random_state=42):
    indices = np.arange(data.num_nodes)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    train_masks = []
    val_masks = []
    test_maks = []

    for train_indices, val_indices in kf.split(indices):
        # Initialize masks for this fold
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        # Set the corresponding indices to True
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        
        # Append the masks for this fold to the lists
        train_masks.append(train_mask)
        val_masks.append(val_mask)

    return train_masks, val_masks


def cross_validate_model(
    data,
    model, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    val_size=0.2,
    optimizer=None,
    criterion=None,
    scheduler=None,
    epochs=500,
    random_state=1, 
    verbose=True,
    print_every=200,
    ):

    train_masks, val_masks = split_data_k_fold(data, n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold_index, (train_mask, val_mask) in enumerate(zip(train_masks, val_masks)):
        
        print(f"Training on fold {fold_index+1}...")

        scores, _, _ = train_evaluate_model(
            data,
            model, 
            device,
            train_mask=train_mask, 
            val_mask=val_mask, 
            val_size=val_size,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=epochs,
            random_state=random_state, 
            verbose=verbose,
            print_every=print_every,
        )

        cv_scores.append(scores)

    return aggregate_cv_scores(cv_scores)


def evaluate_and_visualize_model_performance(
    data,
    model, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    train_mask=None, 
    val_mask=None, 
    val_size=0.2,
    optimizer=None,
    criterion=None,
    scheduler=None,
    epochs=500,
    random_state=1, 
    verbose=True,
    save_plots=True,
    save_path=None,
    print_every=200,
    title_prefix="", 
    test_size=0.2, 
    plot_limit=10**7, 
    plot_bins=50):

    scores, y_val, y_val_pred = train_evaluate_model(
        data=data,
        model=model, 
        device=device,
        train_mask=train_mask, 
        val_mask=val_mask, 
        val_size=val_size,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        random_state=random_state, 
        verbose=verbose,
        print_every=print_every,
    )

    # Determine save path
    if save_plots:
        if save_path is None:
            save_path = os.path.join("output", model.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    
    # Visualization: Predicted vs True Values
    plot_predictions_vs_true(
        y_val, y_val_pred, limit=plot_limit, 
        title=f'{title_prefix}Predicted vs True Values for {model.name}',
        save_image=save_plots, 
        image_path=os.path.join(save_path, f"{model.name}_predictions_vs_true.png") if save_plots else None
    )

    # Visualization: Distribution of Predicted vs True Values
    plot_distribution_difference(
        y_val, y_val_pred, bins=plot_bins, 
        title=f'{title_prefix}Distribution of Predicted vs True Values for {model.name}',
        save_image=save_plots, 
        image_path=os.path.join(save_path, f"{model.name}_distribution_difference.png") if save_plots else None
    )

    return scores