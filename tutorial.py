"""
Complete Tutorial: TabDDPM for Synthetic Data Generation

This tutorial walks through a complete workflow using TabDDPM
to generate synthetic data and evaluate its quality.
"""

import sys

from paths import HERE

sys.path.insert(0, '/mnt/user-data/outputs')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from tabddpm_wrapper import TabDDPM


def tutorial_complete_workflow():
    """
    Complete workflow: Load data → Train TabDDPM → Generate synthetic data → Evaluate
    """
    
    print("=" * 80)
    print("TUTORIAL: Complete TabDDPM Workflow")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Load and Prepare Real Data
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Load and Prepare Data")
    print("-" * 80)
    
    # Load Wine dataset
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    print(f"\nDataset: Wine Recognition")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(wine.feature_names)}")
    print(f"Classes: {len(np.unique(wine.target))}")
    print(f"\nClass distribution:")
    print(df['target'].value_counts().sort_index())
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    # ========================================================================
    # STEP 2: Split Data for Validation
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Split Data into Train/Test Sets")
    print("-" * 80)
    
    # We'll use a small training set to simulate limited data scenario
    train_df, test_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['target']
    )
    
    print(f"\nTraining set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"\nTraining set class distribution:")
    print(train_df['target'].value_counts().sort_index())
    
    # ========================================================================
    # STEP 3: Train Baseline Model on Real Data Only
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Train Baseline Model (Real Data Only)")
    print("-" * 80)
    
    feature_cols = [col for col in df.columns if col != 'target']
    
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(train_df[feature_cols], train_df['target'])
    
    baseline_pred = baseline_model.predict(test_df[feature_cols])
    baseline_acc = accuracy_score(test_df['target'], baseline_pred)
    
    print(f"\n✓ Baseline Model Trained")
    print(f"✓ Test Accuracy: {baseline_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_df['target'], baseline_pred, 
                                target_names=wine.target_names))
    
    # ========================================================================
    # STEP 4: Initialize and Train TabDDPM
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Initialize and Train TabDDPM")
    print("-" * 80)
    
    print("\nInitializing TabDDPM...")
    tabddpm = TabDDPM(
        categorical_columns=[],           # Wine dataset has only numerical features
        target_column='target',
        task_type='multiclass',
        num_timesteps=100,               # Using 100 for faster demo
        gaussian_loss_type='mse',
        scheduler='cosine',
        device='cpu',                    # Change to 'cuda' if GPU available
        seed=42
    )
    
    print("\n✓ TabDDPM initialized")
    print("\nTraining TabDDPM (this may take a few minutes)...")
    
    tabddpm.fit(
        train_df,
        epochs=3000,                     # 3000 epochs for demo, use 5000+ for production
        lr=0.002,
        weight_decay=1e-4,
        batch_size=64,
        verbose=True
    )
    
    print("\n✓ TabDDPM training completed!")
    
    # ========================================================================
    # STEP 5: Generate Synthetic Data
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Generate Synthetic Data")
    print("-" * 80)
    
    print("\nGenerating synthetic samples...")
    
    # Generate same number of samples as training set
    synthetic_df = tabddpm.sample(n_samples=len(train_df), seed=42)
    
    print(f"\n✓ Generated {len(synthetic_df)} synthetic samples")
    print(f"\nSynthetic data class distribution:")
    print(synthetic_df['target'].value_counts().sort_index())
    
    print(f"\nSynthetic data sample:")
    print(synthetic_df.head())
    
    # ========================================================================
    # STEP 6: Compare Real vs Synthetic Data Statistics
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 6: Compare Real vs Synthetic Data Statistics")
    print("-" * 80)
    
    print("\nFeature Statistics Comparison:")
    print("\n" + "=" * 80)
    
    for col in feature_cols[:5]:  # Show first 5 features
        real_mean = train_df[col].mean()
        real_std = train_df[col].std()
        synth_mean = synthetic_df[col].mean()
        synth_std = synthetic_df[col].std()
        
        print(f"\n{col}:")
        print(f"  Real:      Mean={real_mean:>8.3f}, Std={real_std:>8.3f}")
        print(f"  Synthetic: Mean={synth_mean:>8.3f}, Std={synth_std:>8.3f}")
        print(f"  Difference: {abs(real_mean - synth_mean):>8.3f} ({abs(real_mean - synth_mean)/real_mean*100:.1f}%)")
    
    # ========================================================================
    # STEP 7: Train Model on Augmented Data
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 7: Train Model on Augmented Data (Real + Synthetic)")
    print("-" * 80)
    
    # Combine real and synthetic data
    augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    
    print(f"\nAugmented dataset size: {len(augmented_df)}")
    print(f"  - Real samples: {len(train_df)}")
    print(f"  - Synthetic samples: {len(synthetic_df)}")
    
    # Train model on augmented data
    augmented_model = RandomForestClassifier(n_estimators=100, random_state=42)
    augmented_model.fit(augmented_df[feature_cols], augmented_df['target'])
    
    augmented_pred = augmented_model.predict(test_df[feature_cols])
    augmented_acc = accuracy_score(test_df['target'], augmented_pred)
    
    print(f"\n✓ Augmented Model Trained")
    print(f"✓ Test Accuracy: {augmented_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_df['target'], augmented_pred,
                                target_names=wine.target_names))
    
    # ========================================================================
    # STEP 8: Compare Results
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 8: Final Results Comparison")
    print("-" * 80)
    
    print(f"\n{'Model':<30} {'Training Data':<20} {'Test Accuracy':<15}")
    print("=" * 65)
    print(f"{'Baseline':<30} {'Real only':<20} {baseline_acc:<15.4f}")
    print(f"{'With Synthetic Data':<30} {'Real + Synthetic':<20} {augmented_acc:<15.4f}")
    print("=" * 65)
    
    improvement = augmented_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100
    
    if improvement > 0:
        print(f"\n✓ IMPROVEMENT: +{improvement:.4f} ({improvement_pct:+.2f}%)")
    else:
        print(f"\n⚠ Performance change: {improvement:.4f} ({improvement_pct:+.2f}%)")
    
    # ========================================================================
    # STEP 9: Save the Model
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 9: Save TabDDPM Model for Future Use")
    print("-" * 80)
    
    model_path = HERE.joinpath('wine_tabddpm.pt')
    tabddpm.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Demonstrate loading
    print("\nDemonstrating model loading...")
    loaded_model = TabDDPM()
    loaded_model.load(model_path)
    print("✓ Model loaded successfully")
    
    # Generate samples with loaded model
    new_samples = loaded_model.sample(n_samples=100)
    print(f"✓ Generated {len(new_samples)} samples with loaded model")
    
    # ========================================================================
    # STEP 10: Summary and Best Practices
    # ========================================================================
    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETE - KEY TAKEAWAYS")
    print("=" * 80)
    
    print("""
    ✓ Successfully trained TabDDPM on real data
    ✓ Generated high-quality synthetic data
    ✓ Evaluated synthetic data quality
    ✓ Used synthetic data for model improvement
    ✓ Saved model for future use
    
    BEST PRACTICES:
    ──────────────────────────────────────────────────────────────
    1. Data Quality
       • Ensure sufficient real data (>500 samples recommended)
       • Clean and preprocess data before training
       • Check for missing values and outliers
    
    2. Training Configuration
       • Use more epochs (5000-10000) for production
       • Use GPU (device='cuda') for faster training
       • Increase num_timesteps to 1000 for better quality
       • Adjust batch_size based on available memory
    
    3. Evaluation
       • Always compare synthetic vs real data statistics
       • Test downstream model performance
       • Visualize data distributions
       • Check for mode collapse or artifacts
    
    4. Use Cases
       • Data augmentation for small datasets
       • Privacy-preserving data sharing
       • Testing and development environments
       • Handling imbalanced classes
       • Dataset expansion for research
    
    5. When to Use TabDDPM
       ✓ Need high-quality synthetic tabular data
       ✓ Want to preserve complex feature correlations
       ✓ Have mixed numerical and categorical features
       ✓ Need privacy-preserving synthetic data
       ✓ Want to augment small datasets
    """)
    
    print("=" * 80)
    print("For more examples, see: example_usage.py")
    print("For documentation, see: README.md")
    print("=" * 80)


def quick_start_guide():
    """A minimal example to get started quickly"""
    
    print("\n" + "=" * 80)
    print("QUICK START GUIDE - Minimal Example")
    print("=" * 80)
    
    print("""
    # 1. Import and load data
    from tabddpm_wrapper import TabDDPM
    import pandas as pd
    
    df = pd.read_csv('your_data.csv')
    
    # 2. Initialize TabDDPM
    model = TabDDPM(
        categorical_columns=['cat_col1', 'cat_col2'],
        target_column='target',
        device='cuda'  # or 'cpu'
    )
    
    # 3. Train
    model.fit(df, epochs=5000)
    
    # 4. Generate synthetic data
    synthetic_df = model.sample(n_samples=1000)
    
    # 5. Save for later use
    model.save('my_model.pt')
    
    That's it! 🎉
    """)


if __name__ == "__main__":
    # Run the complete tutorial
    tutorial_complete_workflow()
    
    # Show quick start
    quick_start_guide()
    
    print("\n\n" + "=" * 80)
    print("🎉 Tutorial completed successfully!")
    print("=" * 80 + "\n")
