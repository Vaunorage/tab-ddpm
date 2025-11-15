"""
Simple Example: Using TabDDPM Wrapper for Synthetic Data Generation

This script demonstrates how to use the TabDDPM wrapper class
for both classification and regression tasks.
"""

import sys
sys.path.insert(0, '/mnt/user-data/outputs')

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tabddpm_wrapper import TabDDPM


def example_classification():
    """Example: Binary Classification with synthetic data augmentation"""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Binary Classification with Data Augmentation")
    print("=" * 80)
    
    # Create a small dataset
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced classes
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(15)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Class distribution:\n{df['target'].value_counts()}")
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    print(f"\nTraining set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Train baseline model on original data
    print("\n--- Training baseline model on original data ---")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(train_df[feature_names], train_df['target'])
    
    baseline_pred = rf_baseline.predict(test_df[feature_names])
    baseline_acc = accuracy_score(test_df['target'], baseline_pred)
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Train TabDDPM model
    print("\n--- Training TabDDPM model ---")
    tabddpm = TabDDPM(
        categorical_columns=[],
        target_column='target',
        task_type='binclass',
        num_timesteps=100,
        device='cpu'
    )
    
    tabddpm.fit(train_df, epochs=2000, batch_size=128, verbose=True)
    
    # Generate synthetic data
    print("\n--- Generating synthetic data ---")
    synthetic_df = tabddpm.sample(n_samples=1000)
    print(f"Synthetic data size: {len(synthetic_df)}")
    print(f"Synthetic class distribution:\n{synthetic_df['target'].value_counts()}")
    
    # Train model on augmented data (original + synthetic)
    print("\n--- Training model on augmented data ---")
    augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    print(f"Augmented dataset size: {len(augmented_df)}")
    
    rf_augmented = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_augmented.fit(augmented_df[feature_names], augmented_df['target'])
    
    augmented_pred = rf_augmented.predict(test_df[feature_names])
    augmented_acc = accuracy_score(test_df['target'], augmented_pred)
    print(f"Augmented data accuracy: {augmented_acc:.4f}")
    
    print(f"\nImprovement: {(augmented_acc - baseline_acc):.4f}")


def example_regression():
    """Example: Regression task"""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Regression Task")
    print("=" * 80)
    
    # Create regression dataset
    X, y = make_regression(
        n_samples=800,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Target statistics:\n{df['target'].describe()}")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train baseline model
    print("\n--- Training baseline model ---")
    rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_baseline.fit(train_df[feature_names], train_df['target'])
    
    baseline_pred = rf_baseline.predict(test_df[feature_names])
    baseline_rmse = np.sqrt(mean_squared_error(test_df['target'], baseline_pred))
    baseline_r2 = r2_score(test_df['target'], baseline_pred)
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Baseline R²: {baseline_r2:.4f}")
    
    # Train TabDDPM
    print("\n--- Training TabDDPM model ---")
    tabddpm = TabDDPM(
        categorical_columns=[],
        target_column='target',
        task_type='regression',
        num_timesteps=100,
        device='cpu'
    )
    
    tabddpm.fit(train_df, epochs=2000, batch_size=128, verbose=True)
    
    # Generate synthetic data
    print("\n--- Generating synthetic data ---")
    synthetic_df = tabddpm.sample(n_samples=500)
    print(f"Synthetic data size: {len(synthetic_df)}")
    print(f"Synthetic target statistics:\n{synthetic_df['target'].describe()}")
    
    # Save and load model demo
    print("\n--- Testing save/load functionality ---")
    tabddpm.save('/mnt/user-data/outputs/tabddpm_regression.pt')
    
    new_tabddpm = TabDDPM()
    new_tabddpm.load('/mnt/user-data/outputs/tabddpm_regression.pt')
    
    # Generate more samples with loaded model
    more_synthetic_df = new_tabddpm.sample(n_samples=200)
    print(f"Generated {len(more_synthetic_df)} more samples with loaded model")


def example_with_categorical():
    """Example: Mixed numerical and categorical features"""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Mixed Numerical and Categorical Features")
    print("=" * 80)
    
    # Create dataset with categorical features
    np.random.seed(42)
    n_samples = 600
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n_samples),
    })
    
    # Create target based on features (loan approval)
    df['loan_approved'] = (
        (df['credit_score'] > 650) & 
        (df['income'] > 45000)
    ).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    df.loc[noise_idx, 'loan_approved'] = 1 - df.loc[noise_idx, 'loan_approved']
    
    print(f"\nDataset size: {len(df)}")
    print(f"\nFeature types:")
    print(df.dtypes)
    print(f"\nTarget distribution:\n{df['loan_approved'].value_counts()}")
    
    # Train TabDDPM with categorical features
    print("\n--- Training TabDDPM with categorical features ---")
    tabddpm = TabDDPM(
        categorical_columns=['gender', 'education', 'employment'],
        target_column='loan_approved',
        task_type='binclass',
        num_timesteps=100,
        device='cpu'
    )
    
    tabddpm.fit(df, epochs=2000, batch_size=128, verbose=True)
    
    # Generate synthetic data
    print("\n--- Generating synthetic data ---")
    synthetic_df = tabddpm.sample(n_samples=500)
    
    print(f"\nSynthetic data shape: {synthetic_df.shape}")
    print(f"\nSynthetic data sample:")
    print(synthetic_df.head(10))
    
    print(f"\nSynthetic categorical distributions:")
    for col in ['gender', 'education', 'employment']:
        print(f"\n{col}:")
        print(synthetic_df[col].value_counts())


if __name__ == "__main__":
    # Run all examples
    example_classification()
    example_regression()
    example_with_categorical()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
