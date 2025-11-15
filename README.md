# TabDDPM Wrapper - Easy Synthetic Data Generation

A simple, scikit-learn-style wrapper for TabDDPM (Tabular Denoising Diffusion Probabilistic Models) that makes it easy to generate high-quality synthetic tabular data.

## Features

- ✅ **Simple API**: Just `fit()` and `sample()` - works like scikit-learn
- ✅ **Pandas Integration**: Direct support for pandas DataFrames
- ✅ **Automatic Type Detection**: Automatically detects regression vs classification tasks
- ✅ **Mixed Data Types**: Handles both numerical and categorical features
- ✅ **Save/Load Support**: Save trained models and reload them later
- ✅ **GPU Support**: Optional GPU acceleration for faster training

## Installation

### Requirements

```bash
# Core dependencies (from requirements.txt)
torch==1.10.1
numpy==1.21.4
pandas==1.3.4
scikit-learn==1.0.2
scipy==1.7.2
category-encoders==2.3.0
```

### Setup

1. Make sure you have the TabDDPM codebase with all the library files
2. Place `tabddpm_wrapper.py` in your project directory
3. Import and use!

## Quick Start

### Basic Classification Example

```python
from tabddpm_wrapper import TabDDPM
import pandas as pd
from sklearn.datasets import make_classification

# Create your dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# Initialize TabDDPM
model = TabDDPM(
    categorical_columns=[],       # List categorical column names
    target_column='target',       # Name of target column
    num_timesteps=1000,          # Number of diffusion steps
    device='cuda'                # Use 'cuda' or 'cpu'
)

# Train the model
model.fit(df, epochs=5000, batch_size=256)

# Generate synthetic data
synthetic_df = model.sample(n_samples=1000)

print(synthetic_df.head())
```

### Regression Example

```python
from tabddpm_wrapper import TabDDPM
from sklearn.datasets import make_regression

# Create regression dataset
X, y = make_regression(n_samples=800, n_features=10)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# Initialize for regression
model = TabDDPM(
    target_column='target',
    task_type='regression',  # Explicitly set task type
    num_timesteps=1000,
    device='cpu'
)

# Fit and sample
model.fit(df, epochs=5000)
synthetic_df = model.sample(n_samples=500)
```

### Mixed Numerical and Categorical Features

```python
# Your dataset with mixed types
df = pd.DataFrame({
    'age': [25, 30, 35, ...],
    'income': [50000, 60000, 55000, ...],
    'gender': ['M', 'F', 'M', ...],
    'education': ['Bachelor', 'Master', 'PhD', ...],
    'approved': [1, 0, 1, ...]
})

# Specify which columns are categorical
model = TabDDPM(
    categorical_columns=['gender', 'education'],
    target_column='approved',
    num_timesteps=1000,
    device='cuda'
)

model.fit(df, epochs=5000)
synthetic_df = model.sample(n_samples=1000)
```

## API Reference

### Class: `TabDDPM`

#### Initialization Parameters

```python
TabDDPM(
    categorical_columns=None,      # List of categorical column names
    target_column=None,            # Name of target column
    task_type=None,                # 'regression', 'binclass', or 'multiclass'
    num_timesteps=1000,            # Number of diffusion timesteps
    gaussian_loss_type='mse',      # Loss type for continuous features
    scheduler='cosine',            # Noise scheduler ('cosine' or 'linear')
    model_type='mlp',              # Model architecture (currently 'mlp')
    model_params=None,             # Custom model parameters
    device='cuda',                 # Device for training ('cuda' or 'cpu')
    seed=0                         # Random seed
)
```

#### Methods

##### `fit(df, epochs=5000, lr=0.002, weight_decay=1e-4, batch_size=1024, verbose=True)`

Train the TabDDPM model on your data.

**Parameters:**
- `df` (pd.DataFrame): Training data
- `epochs` (int): Number of training steps
- `lr` (float): Learning rate
- `weight_decay` (float): Weight decay for optimizer
- `batch_size` (int): Batch size for training
- `verbose` (bool): Print training progress

**Returns:** `self`

##### `sample(n_samples=1000, batch_size=2000, seed=None, return_numpy=False)`

Generate synthetic samples.

**Parameters:**
- `n_samples` (int): Number of samples to generate
- `batch_size` (int): Batch size for generation
- `seed` (int, optional): Random seed
- `return_numpy` (bool): Return numpy arrays instead of DataFrame

**Returns:** `pd.DataFrame` (or tuple of arrays if `return_numpy=True`)

##### `save(path)`

Save the trained model to disk.

**Parameters:**
- `path` (str): File path to save model

##### `load(path)`

Load a previously saved model.

**Parameters:**
- `path` (str): File path to load model from

**Returns:** `self`

## Advanced Usage

### Custom Model Architecture

```python
model = TabDDPM(
    target_column='target',
    model_params={
        'is_y_cond': True,
        'rtdl_params': {
            'd_layers': [512, 512, 512, 256],  # Hidden layer sizes
            'dropout': 0.1                      # Dropout rate
        }
    }
)
```

### Save and Load Models

```python
# Train and save
model.fit(df, epochs=5000)
model.save('my_tabddpm_model.pt')

# Load and use later
new_model = TabDDPM()
new_model.load('my_tabddpm_model.pt')
synthetic_df = new_model.sample(n_samples=1000)
```

### Data Augmentation for Imbalanced Datasets

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split your data
train_df, test_df = train_test_split(df, test_size=0.2)

# Train TabDDPM on training data
model = TabDDPM(target_column='target')
model.fit(train_df, epochs=5000)

# Generate synthetic samples
synthetic_df = model.sample(n_samples=len(train_df))

# Combine real and synthetic data
augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)

# Train classifier on augmented data
clf = RandomForestClassifier()
clf.fit(augmented_df.drop('target', axis=1), augmented_df['target'])
```

## Performance Tips

### GPU Acceleration

Use GPU for faster training:

```python
model = TabDDPM(device='cuda')
```

### Batch Size

- Larger batch sizes → faster training but more memory
- Typical values: 256-4096
- Adjust based on your GPU memory

### Number of Timesteps

- More timesteps → higher quality but slower
- 100 timesteps: Fast, good for testing
- 1000 timesteps: High quality, recommended for production

### Training Epochs

- Small datasets (< 1000 rows): 3000-5000 epochs
- Medium datasets (1000-10000 rows): 5000-10000 epochs
- Large datasets (> 10000 rows): 10000-20000 epochs

## Comparison with Other Methods

| Method | Quality | Speed | Preserves Correlations | Handles Mixed Types |
|--------|---------|-------|----------------------|-------------------|
| TabDDPM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ Yes | ✅ Yes |
| SMOTE | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ No | ❌ No |
| CTGAN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Yes | ✅ Yes |
| TVAE | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ Partial | ✅ Yes |

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```python
model.fit(df, batch_size=128)  # Instead of 1024
```

### Poor Quality Samples

1. Increase training epochs: `epochs=10000`
2. Increase timesteps: `num_timesteps=1000`
3. Check data preprocessing
4. Ensure sufficient training data (> 500 samples recommended)

### Slow Training

1. Use GPU: `device='cuda'`
2. Increase batch size: `batch_size=4096`
3. Reduce timesteps for testing: `num_timesteps=100`

## Citation

If you use TabDDPM in your research, please cite:

```bibtex
@article{kotelnikov2022tabddpm,
  title={TabDDPM: Modelling Tabular Data with Diffusion Models},
  author={Kotelnikov, Akim and Baranchuk, Dmitry and Rubachev, Ivan and Babenko, Artem},
  journal={arXiv preprint arXiv:2209.15421},
  year={2022}
}
```

## License

This wrapper follows the same license as the original TabDDPM repository.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Examples

See `example_usage.py` for complete working examples including:
- Binary classification
- Regression
- Multi-class classification
- Mixed numerical and categorical features
- Data augmentation workflows

## Support

For issues related to:
- **This wrapper**: Open an issue in this repository
- **TabDDPM core**: Refer to the [original TabDDPM repository](https://github.com/rotot0/tab-ddpm)
