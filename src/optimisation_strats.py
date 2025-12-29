"""
Quick Start Example for Model Optimization Pipeline
====================================================

This script demonstrates different use cases and configurations
for the model optimization pipeline.

Run this after ensuring your data is preprocessed in TFRecords format.
"""

import os
from src.model_optimization import OptimizationConfig, OptimizationPipeline
import pandas as pd
import matplotlib.pyplot as plt
from cste import *


def example_1_quick_test(tfrecord_dir=TFRECORD_OUTPUT_DIR_64):
    """
    Quick test configuration - minimal resources, fast execution.
    Use this to test that everything works before a full run.
    
    Estimated time: 30-60 minutes
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: QUICK TEST")
    print("="*80)
    print("Configuration: Minimal resources, fast execution")
    print("Purpose: Test pipeline functionality")
    print("Estimated time: 30-60 minutes\n")
    
    config = OptimizationConfig(
        # Data
        tfrecord_dir=tfrecord_dir,  # Adjust to your path
        
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


def example_2_standard_run(tfrecord_dir=TFRECORD_OUTPUT_DIR_64):
    """
    Standard configuration - balanced resources and exploration.
    Recommended for most users with 16GB RAM.
    
    Estimated time: 3-6 hours
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
        partial_training_epochs=10,
        full_training_epochs=50,
        early_stopping_patience=5,
        
        # Moderate search
        random_search_iterations=10,
        population_size=8,
        num_generations=5,
        mutation_rate=0.3,
        
        # Output
        output_dir="results_standard",
        
        # Seed
        random_seed=42
    )
    
    pipeline = OptimizationPipeline(config)
    pipeline.run()
    
    print("\n✓ Standard run completed!")
    print(f"Check results in: {config.output_dir}/")


def example_3_thorough_search(tfrecord_dir=TFRECORD_OUTPUT_DIR_64):
    """
    Thorough search configuration - maximum exploration.
    Requires 32GB+ RAM and significant compute time.
    
    Estimated time: 8-15 hours
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
    """
    
    print("\n" + "="*80)
    print("ANALYZING OPTIMIZATION RESULTS")
    print("="*80)
    
    # Load results
    log_path = os.path.join(results_dir, "optimization_log.csv")
    if not os.path.exists(log_path):
        print(f"Error: Results not found at {log_path}")
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
    top_5 = df.nlargest(5, 'val_accuracy')[['model_id', 'generation', 'val_accuracy', 'architecture']]
    print(top_5.to_string(index=False))
    
    # Plot 1: Validation accuracy distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    df['val_accuracy'].hist(bins=20, edgecolor='black')
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Count')
    plt.title('Distribution of Validation Accuracy')
    
    # Plot 2: Evolution progress
    plt.subplot(1, 3, 2)
    df_gen = df.groupby('generation')['val_accuracy'].agg(['max', 'mean'])
    df_gen['max'].plot(label='Best', marker='o')
    df_gen['mean'].plot(label='Average', marker='s')
    plt.xlabel('Generation')
    plt.ylabel('Validation Accuracy')
    plt.title('Evolution Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate vs accuracy
    plt.subplot(1, 3, 3)
    plt.scatter(df['learning_rate'], df['val_accuracy'], alpha=0.6)
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Rate vs Accuracy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "analysis_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {plot_path}")
    plt.close()
    
    # Architecture analysis
    print("\n--- Architecture Analysis ---")
    if 'conv_layers_str' in df.columns:
        df['num_conv_layers'] = df['conv_layers_str'].apply(lambda x: len(eval(x)))
        df['num_dense_layers'] = df['dense_layers_str'].apply(lambda x: len(eval(x)))
        
        print("\nBest accuracy by number of conv layers:")
        print(df.groupby('num_conv_layers')['val_accuracy'].max().to_string())
        
        print("\nBest accuracy by number of dense layers:")
        print(df.groupby('num_dense_layers')['val_accuracy'].max().to_string())
    
    print("\n" + "="*80)


if __name__ == "__main__":
    pass