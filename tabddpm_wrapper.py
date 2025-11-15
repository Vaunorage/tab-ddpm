"""
TabDDPM Wrapper Class

A scikit-learn style wrapper for TabDDPM that works with pandas DataFrames.

Usage:
    from tabddpm_wrapper import TabDDPM
    
    # Initialize
    model = TabDDPM(
        categorical_columns=['cat1', 'cat2'],
        target_column='target',
        num_timesteps=1000,
        device='cuda'
    )
    
    # Fit on your data
    model.fit(df, epochs=5000)
    
    # Generate synthetic data
    synthetic_df = model.sample(n_samples=1000)
"""

import sys
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from data import Dataset, Transformations, transform_dataset, prepare_fast_dataloader
from gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from modules import MLPDiffusion
from util import TaskType

# Add the project root to path to import modules
# Adjust this if running from different location
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class TabDDPM:
    """
    TabDDPM: Tabular Diffusion Denoising Probabilistic Model
    
    A wrapper class that provides a simple interface for training and sampling
    from TabDDPM models using pandas DataFrames.
    
    Parameters
    ----------
    categorical_columns : List[str], optional
        List of column names that are categorical features
    target_column : str, optional
        Name of the target column. If None, assumes unsupervised learning
    task_type : {'regression', 'binclass', 'multiclass'}, optional
        Type of task. If None, will be inferred from target column
    num_timesteps : int, default=1000
        Number of diffusion timesteps
    gaussian_loss_type : {'mse', 'kl'}, default='mse'
        Loss type for Gaussian diffusion
    scheduler : {'cosine', 'linear'}, default='cosine'
        Noise scheduler type
    model_type : {'mlp'}, default='mlp'
        Type of denoising model
    model_params : dict, optional
        Parameters for the model architecture
    device : str, default='cuda'
        Device to use for training ('cuda' or 'cpu')
    seed : int, default=0
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        task_type: Optional[Literal['regression', 'binclass', 'multiclass']] = None,
        num_timesteps: int = 1000,
        gaussian_loss_type: str = 'mse',
        scheduler: str = 'cosine',
        model_type: str = 'mlp',
        model_params: Optional[Dict[str, Any]] = None,
        device: str = 'cuda',
        seed: int = 0
    ):
        self.categorical_columns = categorical_columns or []
        self.target_column = target_column
        self.task_type = task_type
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler
        self.model_type = model_type
        self.seed = seed
        
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available, using CPU instead")
            device = 'cpu'
        self.device = torch.device(device)
        
        # Default model parameters
        if model_params is None:
            self.model_params = {
                'is_y_cond': True,  # Conditional on target
                'rtdl_params': {
                    'd_layers': [256, 256, 256],
                    'dropout': 0.0
                }
            }
        else:
            self.model_params = model_params
            
        # These will be set during fit
        self.model = None
        self.diffusion = None
        self.dataset = None
        self.feature_columns = None
        self.num_classes = None
        self.empirical_class_dist = None
        self.transformations = None
        
        # Preprocessing objects
        self.num_transform = None
        self.cat_transform = None
        self.y_info = {}
        
    def _infer_task_type(self, y: np.ndarray) -> TaskType:
        """Infer task type from target variable"""
        if self.task_type is not None:
            return TaskType(self.task_type)
            
        unique_values = np.unique(y)
        
        # Check if continuous (regression)
        if len(unique_values) > 20 or y.dtype == np.float64:
            return TaskType.REGRESSION
        elif len(unique_values) == 2:
            return TaskType.BINCLASS
        else:
            return TaskType.MULTICLASS
    
    def _prepare_data(self, df: pd.DataFrame) -> Dataset:
        """
        Prepare pandas DataFrame for training
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
            
        Returns
        -------
        Dataset
            Processed dataset ready for training
        """
        df = df.copy()
        
        # Separate features and target
        if self.target_column:
            y = df[self.target_column].values
            X_df = df.drop(columns=[self.target_column])
        else:
            # Unsupervised - create dummy target
            y = np.zeros(len(df))
            X_df = df
            
        self.feature_columns = list(X_df.columns)
        
        # Separate numerical and categorical features
        cat_cols = [col for col in self.categorical_columns if col in X_df.columns]
        num_cols = [col for col in X_df.columns if col not in cat_cols]
        
        # Prepare arrays
        X_num = X_df[num_cols].values.astype(np.float64) if num_cols else None
        X_cat = X_df[cat_cols].values.astype(str) if cat_cols else None
        
        # Infer task type
        task_type = self._infer_task_type(y)
        
        # Determine number of classes
        if task_type == TaskType.REGRESSION:
            self.num_classes = 0
        else:
            self.num_classes = len(np.unique(y))
            
        self.model_params['num_classes'] = self.num_classes
        
        # Create dataset splits (using all data as train for simplicity)
        # In practice, you might want to add validation split
        dataset = Dataset(
            X_num={'train': X_num} if X_num is not None else None,
            X_cat={'train': X_cat} if X_cat is not None else None,
            y={'train': y},
            y_info={},
            task_type=task_type,
            n_classes=self.num_classes if self.num_classes > 0 else None
        )
        
        return dataset
    
    def _get_transformations(self) -> Transformations:
        """Get data transformations configuration"""
        return Transformations(
            seed=self.seed,
            normalization='quantile',
            num_nan_policy=None,
            cat_nan_policy=None,
            cat_min_frequency=None,
            cat_encoding='one-hot' if len(self.categorical_columns) > 0 else None,
            y_policy='default'
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 5000,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        verbose: bool = True
    ):
        """
        Fit the TabDDPM model on a pandas DataFrame
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        epochs : int, default=5000
            Number of training steps
        lr : float, default=0.002
            Learning rate
        weight_decay : float, default=1e-4
            Weight decay for optimizer
        batch_size : int, default=1024
            Batch size for training
        verbose : bool, default=True
            Whether to print training progress
            
        Returns
        -------
        self
            Fitted estimator
        """
        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Prepare data
        if verbose:
            print("Preparing data...")
        raw_dataset = self._prepare_data(df)
        
        # Apply transformations
        self.transformations = self._get_transformations()
        self.dataset = transform_dataset(raw_dataset, self.transformations, None)
        
        # Store preprocessing objects
        if hasattr(self.dataset, 'num_transform'):
            self.num_transform = self.dataset.num_transform
        if hasattr(self.dataset, 'cat_transform'):
            self.cat_transform = self.dataset.cat_transform
        self.y_info = self.dataset.y_info
        
        # Get category sizes for multinomial diffusion
        K = np.array(self.dataset.get_category_sizes('train'))
        if len(K) == 0 or self.transformations.cat_encoding == 'one-hot':
            K = np.array([0])
            
        # Calculate input dimension
        num_numerical_features = (
            self.dataset.X_num['train'].shape[1] 
            if self.dataset.X_num is not None 
            else 0
        )
        d_in = int(np.sum(K) + num_numerical_features)
        self.model_params['d_in'] = d_in
        
        if verbose:
            print(f"Input dimension: {d_in}")
            print(f"Numerical features: {num_numerical_features}")
            print(f"Categorical features: {len(K) if K[0] != 0 else 0}")
            print(f"Number of classes: {self.num_classes}")
        
        # Create model
        if verbose:
            print("Creating model...")
        self.model = MLPDiffusion(**self.model_params)
        self.model.to(self.device)
        
        # Create diffusion process
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device
        )
        self.diffusion.to(self.device)
        self.diffusion.train()
        
        # Store empirical class distribution for sampling
        if self.num_classes > 0:
            _, self.empirical_class_dist = torch.unique(
                torch.from_numpy(self.dataset.y['train']), 
                return_counts=True
            )
            self.empirical_class_dist = self.empirical_class_dist.float()
        else:
            self.empirical_class_dist = torch.ones(1).float()
        
        # Training
        if verbose:
            print("Starting training...")
        self._train(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            verbose=verbose
        )
        
        if verbose:
            print("Training completed!")
            
        return self
    
    def _train(
        self,
        epochs: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        verbose: bool
    ):
        """Internal training loop"""
        # Create data loader
        train_loader = prepare_fast_dataloader(
            self.dataset, 
            split='train', 
            batch_size=batch_size
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Training loop
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_count = 0
        log_every = 100
        print_every = 500
        
        while step < epochs:
            x, y_dict = next(train_loader)
            x = x.to(self.device)
            y_dict = {'y': y_dict['y'].long().to(self.device)}
            
            optimizer.zero_grad()
            
            # Compute loss
            loss_multi, loss_gauss = self.diffusion.mixed_loss(x, y_dict)
            loss = loss_multi + loss_gauss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            curr_count += len(x)
            curr_loss_multi += loss_multi.item() * len(x)
            curr_loss_gauss += loss_gauss.item() * len(x)
            
            if verbose and (step + 1) % log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                
                if (step + 1) % print_every == 0:
                    print(f'Step {(step + 1)}/{epochs} | '
                          f'Multinomial Loss: {mloss:.4f} | '
                          f'Gaussian Loss: {gloss:.4f} | '
                          f'Total: {mloss + gloss:.4f}')
                
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0
            
            # Learning rate annealing
            frac_done = step / epochs
            current_lr = lr * (1 - frac_done)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            
            step += 1
    
    def sample(
        self,
        n_samples: int = 1000,
        batch_size: int = 2000,
        seed: Optional[int] = None,
        return_numpy: bool = False
    ) -> pd.DataFrame:
        """
        Generate synthetic samples
        
        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate
        batch_size : int, default=2000
            Batch size for sampling
        seed : int, optional
            Random seed for sampling
        return_numpy : bool, default=False
            If True, return numpy arrays instead of DataFrame
            
        Returns
        -------
        pd.DataFrame or tuple
            Generated synthetic data as DataFrame, or (X_num, X_cat, y) if return_numpy=True
        """
        if self.model is None or self.diffusion is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.diffusion.eval()
        
        with torch.no_grad():
            # Sample from diffusion model
            X_gen, y_gen = self.diffusion.sample_all(
                n_samples, 
                batch_size, 
                self.empirical_class_dist.to(self.device),
                ddim=False
            )
        
        X_gen = X_gen.cpu().numpy()
        y_gen = y_gen.cpu().numpy()
        
        # Get number of numerical features
        num_numerical_features = (
            self.dataset.X_num['train'].shape[1] 
            if self.dataset.X_num is not None 
            else 0
        )
        
        # Separate numerical and categorical
        X_num_gen = X_gen[:, :num_numerical_features] if num_numerical_features > 0 else None
        X_cat_gen = X_gen[:, num_numerical_features:] if X_gen.shape[1] > num_numerical_features else None
        
        # Inverse transform numerical features
        if X_num_gen is not None and self.num_transform is not None:
            X_num_gen = self.num_transform.inverse_transform(X_num_gen)
        
        # Inverse transform categorical features
        if X_cat_gen is not None and self.cat_transform is not None:
            # For one-hot encoding, convert back to categories
            if self.transformations.cat_encoding == 'one-hot':
                # Convert probabilities to one-hot
                def to_good_ohe(ohe_encoder, X):
                    indices = np.cumsum([0] + list(ohe_encoder._n_features_outs))
                    X_res = []
                    for i in range(1, len(indices)):
                        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
                        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
                        X_res.append(np.where(t >= 0, 1, 0))
                    return np.hstack(X_res)
                
                X_cat_gen = to_good_ohe(
                    self.cat_transform.steps[0][1], 
                    X_cat_gen
                )
            
            X_cat_gen = self.cat_transform.inverse_transform(X_cat_gen)
        
        # Inverse transform target
        if self.y_info and 'mean' in self.y_info and 'std' in self.y_info:
            # Regression: denormalize
            y_gen = y_gen * self.y_info['std'] + self.y_info['mean']
        elif self.y_info and 'label_encoder' in self.y_info:
            # Classification: inverse label encoding
            y_gen = self.y_info['label_encoder'].inverse_transform(y_gen.astype(int))
        
        if return_numpy:
            return X_num_gen, X_cat_gen, y_gen
        
        # Create DataFrame
        data_dict = {}
        
        # Add numerical features
        if X_num_gen is not None:
            num_cols = [col for col in self.feature_columns 
                       if col not in self.categorical_columns]
            for i, col in enumerate(num_cols):
                data_dict[col] = X_num_gen[:, i]
        
        # Add categorical features
        if X_cat_gen is not None:
            cat_cols = [col for col in self.feature_columns 
                       if col in self.categorical_columns]
            for i, col in enumerate(cat_cols):
                data_dict[col] = X_cat_gen[:, i]
        
        # Add target
        if self.target_column:
            data_dict[self.target_column] = y_gen
        
        return pd.DataFrame(data_dict)
    
    def save(self, path: str):
        """
        Save the trained model
        
        Parameters
        ----------
        path : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Call fit() first.")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'categorical_columns': self.categorical_columns,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'num_classes': self.num_classes,
            'empirical_class_dist': self.empirical_class_dist,
            'transformations': self.transformations,
            'y_info': self.y_info,
            'num_transform': self.num_transform,
            'cat_transform': self.cat_transform,
            'task_type': str(self.dataset.task_type) if self.dataset else None,
            'config': {
                'num_timesteps': self.num_timesteps,
                'gaussian_loss_type': self.gaussian_loss_type,
                'scheduler': self.scheduler,
                'model_type': self.model_type,
                'seed': self.seed
            }
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load a trained model
        
        Parameters
        ----------
        path : str
            Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore configuration
        self.model_params = checkpoint['model_params']
        self.categorical_columns = checkpoint['categorical_columns']
        self.target_column = checkpoint['target_column']
        self.feature_columns = checkpoint['feature_columns']
        self.num_classes = checkpoint['num_classes']
        self.empirical_class_dist = checkpoint['empirical_class_dist']
        self.transformations = checkpoint['transformations']
        self.y_info = checkpoint['y_info']
        self.num_transform = checkpoint['num_transform']
        self.cat_transform = checkpoint['cat_transform']
        
        config = checkpoint['config']
        self.num_timesteps = config['num_timesteps']
        self.gaussian_loss_type = config['gaussian_loss_type']
        self.scheduler = config['scheduler']
        self.model_type = config['model_type']
        self.seed = config['seed']
        
        # Recreate model
        self.model = MLPDiffusion(**self.model_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Recreate diffusion
        K = np.array([0])  # This will be updated from dataset
        num_numerical_features = self.model_params['d_in']  # Approximate
        
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device
        )
        self.diffusion.to(self.device)
        self.diffusion.eval()
        
        print(f"Model loaded from {path}")
        
        return self


def plot_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot distribution comparisons between real and synthetic data.
    
    Parameters
    ----------
    real_df : pd.DataFrame
        Real data
    synthetic_df : pd.DataFrame
        Synthetic data
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    """
    columns = real_df.columns.tolist()
    n_cols = len(columns)
    
    # Calculate grid dimensions
    n_rows = int(np.ceil(n_cols / 3))
    n_plot_cols = min(3, n_cols)
    
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(5 * n_plot_cols, 4 * n_rows))
    
    # Handle single subplot case
    if n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Check if column is numeric or categorical
        if pd.api.types.is_numeric_dtype(real_df[col]):
            # For numeric columns, use histograms
            ax.hist(real_df[col].dropna(), bins=30, alpha=0.5, label='Real', density=True, color='blue')
            ax.hist(synthetic_df[col].dropna(), bins=30, alpha=0.5, label='Synthetic', density=True, color='orange')
            ax.set_ylabel('Density')
        else:
            # For categorical columns, use bar plots
            real_counts = real_df[col].value_counts(normalize=True).sort_index()
            synth_counts = synthetic_df[col].value_counts(normalize=True).sort_index()
            
            x = np.arange(len(real_counts))
            width = 0.35
            
            ax.bar(x - width/2, real_counts.values, width, label='Real', alpha=0.7, color='blue')
            ax.bar(x + width/2, synth_counts.values, width, label='Synthetic', alpha=0.7, color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(real_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Proportion')
        
        ax.set_xlabel(col)
        ax.set_title(f'Distribution: {col}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_classification

    print("=" * 80)
    print("TabDDPM Example - Classification")
    print("=" * 80)
    
    # Classification example
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    # Initialize and train
    model = TabDDPM(
        categorical_columns=[],
        target_column='target',
        task_type='binclass',
        num_timesteps=100,  # Reduced for quick demo
        device='cpu'  # Use 'cuda' if available
    )
    
    print("\nTraining model...")
    model.fit(df, epochs=1000, batch_size=128, verbose=True)
    
    # Generate synthetic data
    print("\nGenerating synthetic samples...")
    synthetic_df = model.sample(n_samples=500)
    
    print("\nOriginal data shape:", df.shape)
    print("Synthetic data shape:", synthetic_df.shape)
    print("\nOriginal data head:")
    print(df.head())
    print("\nSynthetic data head:")
    print(synthetic_df.head())
    
    # Save model
    model.save('tabddpm_model.pt')
    
    # Load model
    new_model = TabDDPM()
    new_model.load('tabddpm_model.pt')
    
    # Plot distributions
    print("\nPlotting distribution comparisons...")
    plot_distributions(df, synthetic_df, save_path='distribution_comparison.png')
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
