"""
Quick Start Example for Model Optimization Pipeline
====================================================

This script demonstrates different use cases and configurations
for the model optimization pipeline.

Run this after ensuring your data is preprocessed in TFRecords format.

VERSION: 2.0 - Compatible with enhanced pipeline
"""

import os
from src.model_optimization import OptimizationConfig, OptimizationPipeline
import pandas as pd
import matplotlib.pyplot as plt
from cste import *


def example_0_pipeline_test(tfrecord_dir=TFRECORD_OUTPUT_DIR_64):
    """
    Basic pipeline test - minimal configuration.
    Use this to verify that the optimization pipeline runs end-to-end.
    
    Estimated time: 15-30 minutes on a standard machine with GPU, and 16GB RAM.
    
    All models are saved to:
    - results_pipeline_test/models/random_search_models/*.keras
    - results_pipeline_test/models/neuro_evolution_models/*.keras
    - results_pipeline_test/models/final_model.keras
    
    Performance tracking in: results_pipeline_test/models_perf.csv
    """
    print("\n" + "="*80)
    print("EXAMPLE 0: PIPELINE TEST")
    print("="*80)
    print("Configuration: Minimal resources, fast execution")
    print("Purpose: Test pipeline functionality")
    print("Estimated time: 15-30 minutes\n")
    
    config = OptimizationConfig(
        # Data
        tfrecord_dir=tfrecord_dir,
        
        # Fast training
        batch_size=32,
        partial_training_epochs=1,      # Very few epochs
        full_training_epochs=2,        # Shorter final training
        
        # Minimal search
        random_search_iterations=2,     # Just 2 random configs
        population_size=2,              # Small population
        num_generations=1,              # Only 1 generation
        
        # Output
        output_dir="results_pipeline_test",
        
        # Seed
        random_seed=42
    )
    
    pipeline = OptimizationPipeline(config)
    pipeline.run()
    
    print("\n✓ Pipeline test completed!")
    print(f"Check results in: {config.output_dir}/")
    print(f"  - Models: {config.output_dir}/models/")
    print(f"  - Performance CSV: {config.output_dir}/models_perf.csv")
    print(f"  - Report: {config.output_dir}/exec_report/optimization_summary.txt")


def example_1_quick_run(tfrecord_dir=TFRECORD_OUTPUT_DIR_64):
    """
    Quick test configuration - minimal resources, fast execution.
    Use this to test that everything works before a full run.
    
    Estimated time: 30-60 minutes on a standard machine with GPU, and 16GB RAM.
    
    All models are saved to:
    - results_quick_test/models/random_search_models/*.keras
    - results_quick_test/models/neuro_evolution_models/*.keras
    - results_quick_test/models/final_model.keras
    
    Performance tracking in: results_quick_test/models_perf.csv
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: QUICK TEST")
    print("="*80)
    print("Configuration: Minimal resources, fast execution")
    print("Purpose: Test pipeline functionality")
    print("Estimated time: 30-60 minutes\n")
    
    config = OptimizationConfig(
        # Data
        tfrecord_dir=tfrecord_dir,
        
        # Fast training
        batch_size=32,
        partial_training_epochs=5,      # Very few epochs
        full_training_epochs=20,        # Shorter final training
        
        # Minimal search
        random_search_iterations=3,     # Just 3 random configs
        population_size=4,              # Small population
        num_generations=2,              # Only 2 generations
        
        # Output
        output_dir="results_quick_test",
        
        # Seed
        random_seed=42
    )
    
    pipeline = OptimizationPipeline(config)
    pipeline.run()
    
    print("\n✓ Quick test completed!")
    print(f"Check results in: {config.output_dir}/")
    print(f"  - Models: {config.output_dir}/models/")
    print(f"  - Performance CSV: {config.output_dir}/models_perf.csv")
    print(f"  - Report: {config.output_dir}/exec_report/optimization_summary.txt")


def example_2_standard_run(tfrecord_dir=TFRECORD_OUTPUT_DIR_64):
    """
    Standard configuration - balanced resources and exploration.
    Recommended for most users with 16GB RAM.
    
    Estimated time: 3-6 hours
    
    This will create:
    - 10 random search models
    - 8 models per generation × 5 generations = 40 neuroevolution models
    - 1 final model
    Total: 51 models saved
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: STANDARD RUN")
    print("="*80)
    print("Configuration: Balanced resources (16GB RAM)")
    print("Purpose: Good exploration with reasonable compute")
    print("Estimated time: 3-6 hours\n")
    
    config = OptimizationConfig(
        # Data
        tfrecord_dir=tfrecord_dir,
        
        # Standard training
        batch_size=32,
        partial_training_epochs=7,
        full_training_epochs=30,
        early_stopping_patience=3,
        
        # Moderate search
        random_search_iterations=7,
        population_size=4,
        num_generations=3,
        mutation_rate=0.35,
        
        # Output
        output_dir="results_standard",
        
        # Seed
        random_seed=42
    )
    
    pipeline = OptimizationPipeline(config)
    pipeline.run()
    
    print("\n✓ Standard run completed!")
    print(f"Check results in: {config.output_dir}/")
    print(f"  - {config.random_search_iterations} random search models")
    print(f"  - {config.population_size * config.num_generations} neuroevolution models")
    print(f"  - 1 final model")


def example_3_thorough_search(tfrecord_dir=TFRECORD_OUTPUT_DIR_64):
    """
    Thorough search configuration - maximum exploration.
    Requires 32GB+ RAM and significant compute time.
    
    Estimated time: 8-15 hours

    This will create:
    - 20 random search models
    - 16 models per generation × 10 generations = 160 neuroevolution models
    - 1 final model
    Total: 181 models saved

    !Note that this function has not been tested, because of resource limitations.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: THOROUGH SEARCH")
    print("="*80)
    print("Configuration: Maximum exploration (32GB+ RAM)")
    print("Purpose: Find best possible architecture")
    print("Estimated time: 8-15 hours\n")
    
    config = OptimizationConfig(
        # Data
        tfrecord_dir=tfrecord_dir,
        
        # Longer training
        batch_size=32,
        partial_training_epochs=15,     # More epochs for better evaluation
        full_training_epochs=100,       # Full training
        early_stopping_patience=10,
        
        # Extensive search
        random_search_iterations=20,    # More random exploration
        population_size=16,             # Larger population
        num_generations=10,             # More generations
        mutation_rate=0.3,
        
        # Output
        output_dir="results_thorough",
        
        # Seed
        random_seed=42
    )
    
    pipeline = OptimizationPipeline(config)
    pipeline.run()
    
    print("\n✓ Thorough search completed!")
    print(f"Check results in: {config.output_dir}/")


def analyze_results(results_dir="results_standard"):
    """
    Analyze optimization results and generate visualizations.
    
    Args:
        results_dir: Directory containing optimization results
    
    This function now uses the new models_perf.csv file.
    """
    
    print("\n" + "="*80)
    print("ANALYZING OPTIMIZATION RESULTS")
    print("="*80)
    
    # Load results from new CSV location
    log_path = os.path.join(results_dir, "models_perf.csv")
    if not os.path.exists(log_path):
        print(f"Error: Results not found at {log_path}")
        print(f"Expected file: models_perf.csv")
        return
    
    df = pd.read_csv(log_path)
    print(f"\nLoaded {len(df)} model evaluations")
    
    # Basic statistics
    print("\n--- Performance Statistics ---")
    print(f"Best validation accuracy: {df['val_accuracy'].max():.4f}")
    print(f"Average validation accuracy: {df['val_accuracy'].mean():.4f}")
    print(f"Std validation accuracy: {df['val_accuracy'].std():.4f}")
    
    # Best models
    print("\n--- Top 5 Models ---")
    top_5 = df.nlargest(5, 'val_accuracy')[['model_id', 'generation', 'val_accuracy', 'architecture', 'model_path']]
    print(top_5.to_string(index=False))
    
    # Check saved models
    print("\n--- Saved Models ---")
    print(f"Random search models: {len(df[df['generation'] == 0])}")
    print(f"Neuroevolution models: {len(df[df['generation'] > 0])}")
    
    if 'model_path' in df.columns:
        saved_models = df[df['model_path'].notna()]
        print(f"Total models with saved paths: {len(saved_models)}")
        
        # Verify files exist
        existing = saved_models['model_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)
        print(f"Models verified on disk: {existing.sum()}")
    
    # Plot 1: Validation accuracy distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    df['val_accuracy'].hist(bins=20, edgecolor='black')
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Count')
    plt.title('Distribution of Validation Accuracy')
    
    # Plot 2: Evolution progress
    plt.subplot(1, 4, 2)
    df_gen = df.groupby('generation')['val_accuracy'].agg(['max', 'mean'])
    df_gen['max'].plot(label='Best', marker='o')
    df_gen['mean'].plot(label='Average', marker='s')
    plt.xlabel('Generation')
    plt.ylabel('Validation Accuracy')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate vs accuracy
    plt.subplot(1, 4, 3)
    plt.scatter(df['learning_rate'], df['val_accuracy'], alpha=0.6)
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Rate vs Accuracy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Models per generation
    plt.subplot(1, 4, 4)
    gen_counts = df['generation'].value_counts().sort_index()
    gen_counts.plot(kind='bar')
    plt.xlabel('Generation')
    plt.ylabel('Number of Models')
    plt.title('Models per Generation')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "analysis_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {plot_path}")
    plt.close()
    
    # Architecture analysis
    print("\n--- Architecture Analysis ---")
    if 'conv_layers' in df.columns:
        df['num_conv_layers'] = df['conv_layers'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)
        df['num_dense_layers'] = df['dense_layers'].apply(lambda x: len(eval(x)) if pd.notna(x) else 0)
        
        print("\nBest accuracy by number of conv layers:")
        print(df.groupby('num_conv_layers')['val_accuracy'].max().to_string())
        
        print("\nBest accuracy by number of dense layers:")
        print(df.groupby('num_dense_layers')['val_accuracy'].max().to_string())
    
    # Save analysis summary
    summary_path = os.path.join(results_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("OPTIMIZATION RESULTS ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total models evaluated: {len(df)}\n")
        f.write(f"Best validation accuracy: {df['val_accuracy'].max():.4f}\n")
        f.write(f"Average validation accuracy: {df['val_accuracy'].mean():.4f}\n")
        f.write(f"Std validation accuracy: {df['val_accuracy'].std():.4f}\n\n")
        
        f.write("Top 5 Models:\n")
        f.write(top_5.to_string(index=False))
        f.write("\n\n")
        
        f.write("Models by Generation:\n")
        f.write(df['generation'].value_counts().sort_index().to_string())
    
    print(f"\n✓ Analysis summary saved to: {summary_path}")
    print("\n" + "="*80)


def verify_saved_models(results_dir="results_standard"):
    """
    Verify that all models in the CSV are actually saved on disk.
    
    Args:
        results_dir: Directory containing optimization results
    """
    print("\n" + "="*80)
    print("VERIFYING SAVED MODELS")
    print("="*80)
    
    log_path = os.path.join(results_dir, "models_perf.csv")
    if not os.path.exists(log_path):
        print(f"Error: CSV not found at {log_path}")
        return
    
    df = pd.read_csv(log_path)
    
    print(f"\nTotal models in CSV: {len(df)}")
    print(f"Models with path specified: {df['model_path'].notna().sum()}")
    
    # Check each model
    missing = []
    for idx, row in df.iterrows():
        if pd.notna(row['model_path']):
            if not os.path.exists(row['model_path']):
                missing.append({
                    'model_id': row['model_id'],
                    'path': row['model_path']
                })
    
    if missing:
        print(f"\n⚠️  WARNING: {len(missing)} models missing from disk:")
        for m in missing[:10]:  # Show first 10
            print(f"  - {m['model_id']}: {m['path']}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("\n✓ All models verified on disk!")
    
    # Calculate total size
    total_size = 0
    for idx, row in df.iterrows():
        if pd.notna(row['model_path']) and os.path.exists(row['model_path']):
            total_size += os.path.getsize(row['model_path'])
    
    print(f"\nTotal disk space used: {total_size / (1024**2):.2f} MB")
    print("="*80)


if __name__ == "__main__":
    # Example usage
    print("Model Optimization Pipeline - Examples")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. example_1_quick_test() - Fast test (30-60 min)")
    print("  2. example_2_standard_run() - Standard run (3-6 hours)")
    print("  3. example_3_thorough_search() - Thorough search (8-15 hours)")
    print("\nAnalysis functions:")
    print("  - analyze_results('results_dir') - Generate plots and statistics")
    print("  - verify_saved_models('results_dir') - Verify all models are saved")
    print("\n" + "="*80)