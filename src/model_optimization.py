"""
Automatic Model Optimization Pipeline for Music Genre Classification
=====================================================================

This script implements a hybrid optimization strategy combining:
1. Random/Grid Search for baseline exploration
2. NeuroEvolution for iterative architecture refinement
3. Comprehensive evaluation and logging

Designed for limited computational resources (16GB RAM).

Author: AI Assistant
Date: 2025-12-27
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from dataclasses import dataclass, asdict
import copy

# Import project modules
from src.cste import *
from src.logger import get_logger
from model_generator import build_and_compile_model_03
from model_training import train_model_pipeline_04, evaluate_model_on_test
from src.cste import *

log = get_logger("model_optimization")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline."""
    
    # Data paths
    tfrecord_dir: str = TFRECORD_OUTPUT_DIR
    
    # Resource constraints
    batch_size: int = 32  # Moderate batch size for 16GB RAM
    partial_training_epochs: int = 10  # Few epochs for quick evaluation
    full_training_epochs: int = 50  # Full training for best model
    early_stopping_patience: int = 5
    
    # Random/Grid Search
    random_search_iterations: int = 10  # Number of random configurations to try
    
    # NeuroEvolution
    population_size: int = 8  # Number of models per generation (4-16)
    num_generations: int = 5  # Number of evolution cycles
    mutation_rate: float = 0.3  # Probability of mutation
    selection_ratio: float = 0.5  # Top 50% survive to next generation
    
    # Output paths
    output_dir: str = "optimization_results"
    models_dir: str = "optimization_results/models"
    logs_dir: str = "optimization_results/logs"
    
    # Model registry
    optimization_log_csv: str = "optimization_results/optimization_log.csv"
    generation_log_csv: str = "optimization_results/generation_log.csv"
    
    # Random seed
    random_seed: int = 42


# ============================================================================
# HYPERPARAMETER SPACE DEFINITION
# ============================================================================

class HyperparameterSpace:
    # check documents/research_range
    """Defines the search space for model hyperparameters."""
    
    # Convolutional layer options
    CONV_FILTERS = [16, 32, 64, 128]
    KERNEL_SIZES = [(3, 3), (5, 5), (7, 7)]
    POOL_SIZES = [(2, 2), (3, 3)]
    CONV_ACTIVATIONS = ['relu', 'elu', 'selu']
    
    # Dense layer options
    DENSE_NEURONS = [64, 128, 256, 512]
    DENSE_ACTIVATIONS = ['relu', 'elu', 'selu']
    
    # Regularization
    DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Training hyperparameters
    LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005]
    
    # Architecture depth
    MIN_CONV_LAYERS = 2
    MAX_CONV_LAYERS = 4
    MIN_DENSE_LAYERS = 1
    MAX_DENSE_LAYERS = 3
    
    @staticmethod
    def sample_architecture() -> Dict:
        """Sample a random architecture configuration."""
        num_conv_layers = random.randint(
            HyperparameterSpace.MIN_CONV_LAYERS,
            HyperparameterSpace.MAX_CONV_LAYERS
        )
        num_dense_layers = random.randint(
            HyperparameterSpace.MIN_DENSE_LAYERS,
            HyperparameterSpace.MAX_DENSE_LAYERS
        )
        
        conv_layers = [
            (random.choice(HyperparameterSpace.CONV_FILTERS),
             random.choice(HyperparameterSpace.KERNEL_SIZES))
            for _ in range(num_conv_layers)
        ]
        
        conv_activations = [
            random.choice(HyperparameterSpace.CONV_ACTIVATIONS)
            for _ in range(num_conv_layers)
        ]
        
        dense_layers = [
            random.choice(HyperparameterSpace.DENSE_NEURONS)
            for _ in range(num_dense_layers)
        ]
        
        dense_activations = [
            random.choice(HyperparameterSpace.DENSE_ACTIVATIONS)
            for _ in range(num_dense_layers)
        ]
        
        dropout_rates = [
            random.choice(HyperparameterSpace.DROPOUT_RATES)
            for _ in range(num_conv_layers)
        ]
        
        return {
            'conv_layers': conv_layers,
            'conv_activations': conv_activations,
            'pool_size': random.choice(HyperparameterSpace.POOL_SIZES),
            'dense_layers': dense_layers,
            'dense_activations': dense_activations,
            'dropout_rates': dropout_rates,
            'learning_rate': random.choice(HyperparameterSpace.LEARNING_RATES)
        }


# ============================================================================
# MODEL REPRESENTATION
# ============================================================================

@dataclass
class ModelConfig:
    """Represents a complete model configuration."""
    
    # Architecture
    conv_layers: List[Tuple[int, Tuple[int, int]]]
    conv_activations: List[str]
    pool_size: Tuple[int, int]
    dense_layers: List[int]
    dense_activations: List[str]
    dropout_rates: List[float]
    
    # Training
    learning_rate: float
    
    # Metadata
    generation: int = 0
    parent_id: Optional[str] = None
    model_id: Optional[str] = None
    
    # Performance metrics
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    train_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_loss: Optional[float] = None
    epochs_trained: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return asdict(self)
    
    def get_architecture_string(self) -> str:
        """Get human-readable architecture description."""
        conv_desc = "-".join([f"C{f}k{k}" for f, k in self.conv_layers])
        dense_desc = "-".join([f"D{n}" for n in self.dense_layers])
        return f"{conv_desc}_{dense_desc}"


# ============================================================================
# MUTATION OPERATORS
# ============================================================================

class MutationOperators:
    """Implements various mutation strategies for neuroevolution."""
    
    @staticmethod
    def mutate_learning_rate(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Mutate learning rate (small change, high probability)."""
        if random.random() < mutation_rate:
            # Multiply by random factor between 0.5 and 2.0
            factor = random.uniform(0.5, 2.0)
            config.learning_rate = np.clip(
                config.learning_rate * factor,
                0.00001,
                0.01
            )
            log.debug(f"Mutated learning rate to {config.learning_rate}")
        return config
    
    @staticmethod
    def mutate_dropout(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Mutate dropout rates (small change, high probability)."""
        if random.random() < mutation_rate:
            idx = random.randint(0, len(config.dropout_rates) - 1)
            config.dropout_rates[idx] = random.choice(HyperparameterSpace.DROPOUT_RATES)
            log.debug(f"Mutated dropout at layer {idx} to {config.dropout_rates[idx]}")
        return config
    
    @staticmethod
    def mutate_conv_filters(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Mutate number of filters in conv layer (medium change)."""
        if random.random() < mutation_rate * 0.7:
            idx = random.randint(0, len(config.conv_layers) - 1)
            filters, kernel = config.conv_layers[idx]
            new_filters = random.choice(HyperparameterSpace.CONV_FILTERS)
            config.conv_layers[idx] = (new_filters, kernel)
            log.debug(f"Mutated conv filters at layer {idx} to {new_filters}")
        return config
    
    @staticmethod
    def mutate_kernel_size(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Mutate kernel size (medium change)."""
        if random.random() < mutation_rate * 0.7:
            idx = random.randint(0, len(config.conv_layers) - 1)
            filters, kernel = config.conv_layers[idx]
            new_kernel = random.choice(HyperparameterSpace.KERNEL_SIZES)
            config.conv_layers[idx] = (filters, new_kernel)
            log.debug(f"Mutated kernel size at layer {idx} to {new_kernel}")
        return config
    
    @staticmethod
    def mutate_dense_neurons(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Mutate number of neurons in dense layer (medium change)."""
        if random.random() < mutation_rate * 0.7:
            idx = random.randint(0, len(config.dense_layers) - 1)
            config.dense_layers[idx] = random.choice(HyperparameterSpace.DENSE_NEURONS)
            log.debug(f"Mutated dense neurons at layer {idx} to {config.dense_layers[idx]}")
        return config
    
    @staticmethod
    def add_conv_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Add a convolutional layer (large change, low probability)."""
        if (random.random() < mutation_rate * 0.3 and 
            len(config.conv_layers) < HyperparameterSpace.MAX_CONV_LAYERS):
            
            new_layer = (
                random.choice(HyperparameterSpace.CONV_FILTERS),
                random.choice(HyperparameterSpace.KERNEL_SIZES)
            )
            config.conv_layers.append(new_layer)
            config.conv_activations.append(random.choice(HyperparameterSpace.CONV_ACTIVATIONS))
            config.dropout_rates.append(random.choice(HyperparameterSpace.DROPOUT_RATES))
            log.debug(f"Added conv layer: {new_layer}")
        return config
    
    @staticmethod
    def remove_conv_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Remove a convolutional layer (large change, low probability)."""
        if (random.random() < mutation_rate * 0.2 and 
            len(config.conv_layers) > HyperparameterSpace.MIN_CONV_LAYERS):
            
            idx = random.randint(0, len(config.conv_layers) - 1)
            config.conv_layers.pop(idx)
            config.conv_activations.pop(idx)
            config.dropout_rates.pop(idx)
            log.debug(f"Removed conv layer at index {idx}")
        return config
    
    @staticmethod
    def add_dense_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Add a dense layer (large change, low probability)."""
        if (random.random() < mutation_rate * 0.3 and 
            len(config.dense_layers) < HyperparameterSpace.MAX_DENSE_LAYERS):
            
            new_neurons = random.choice(HyperparameterSpace.DENSE_NEURONS)
            config.dense_layers.append(new_neurons)
            config.dense_activations.append(random.choice(HyperparameterSpace.DENSE_ACTIVATIONS))
            log.debug(f"Added dense layer: {new_neurons} neurons")
        return config
    
    @staticmethod
    def remove_dense_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Remove a dense layer (large change, low probability)."""
        if (random.random() < mutation_rate * 0.2 and 
            len(config.dense_layers) > HyperparameterSpace.MIN_DENSE_LAYERS):
            
            idx = random.randint(0, len(config.dense_layers) - 1)
            config.dense_layers.pop(idx)
            config.dense_activations.pop(idx)
            log.debug(f"Removed dense layer at index {idx}")
        return config
    
    @staticmethod
    def mutate(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Apply all mutation operators to a config."""
        # Create a deep copy to avoid modifying original
        mutated = copy.deepcopy(config)
        
        # Apply mutations with different probabilities
        mutated = MutationOperators.mutate_learning_rate(mutated, mutation_rate)
        mutated = MutationOperators.mutate_dropout(mutated, mutation_rate)
        mutated = MutationOperators.mutate_conv_filters(mutated, mutation_rate)
        mutated = MutationOperators.mutate_kernel_size(mutated, mutation_rate)
        mutated = MutationOperators.mutate_dense_neurons(mutated, mutation_rate)
        mutated = MutationOperators.add_conv_layer(mutated, mutation_rate)
        mutated = MutationOperators.remove_conv_layer(mutated, mutation_rate)
        mutated = MutationOperators.add_dense_layer(mutated, mutation_rate)
        mutated = MutationOperators.remove_dense_layer(mutated, mutation_rate)
        
        return mutated


# ============================================================================
# MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """Handles model training and evaluation."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def evaluate_model(
        self,
        model_config: ModelConfig,
        epochs: int,
        model_name: str,
        save_model: bool = False
    ) -> ModelConfig:
        """
        Train and evaluate a model configuration.
        
        Args:
            model_config: Model configuration to evaluate
            epochs: Number of training epochs
            model_name: Unique name for this model
            save_model: Whether to save the trained model
        
        Returns:
            Updated model_config with performance metrics
        """
        log.info(f"Evaluating model: {model_name}")
        log.info(f"Architecture: {model_config.get_architecture_string()}")
        
        try:
            # Build model
            model = build_and_compile_model_03(
                model_name=model_name,
                input_shape=ModelDefaults.INPUT_SHAPE,
                output_units=ModelDefaults.OUTPUT_UNITS,
                conv_layers=model_config.conv_layers,
                conv_activations=model_config.conv_activations,
                pool_size=model_config.pool_size,
                dense_layers=model_config.dense_layers,
                dense_activations=model_config.dense_activations,
                dropout_rates=model_config.dropout_rates,
                optimizer=ModelDefaults.OPTIMIZER,
                learning_rate=model_config.learning_rate,
                loss=ModelDefaults.LOSS,
                metrics=ModelDefaults.METRICS,
                save=False  # Don't save to registry yet
            )
            
            # Train model
            trained_model, history = train_model_pipeline_04(
                model=model,
                tfrecord_dir=self.config.tfrecord_dir,
                batch_size=self.config.batch_size,
                epochs=epochs,
                learning_rate=model_config.learning_rate,
                early_stopping_patience=self.config.early_stopping_patience,
                save=save_model,
                model_save_dir=self.config.models_dir if save_model else None,
                model_registry_csv=self.config.optimization_log_csv if save_model else None,
                notes=f"Generation {model_config.generation}",
                cache_dataset=False,
                shuffle_buffer_size=1000  # Reduced for memory
            )
            
            # Extract metrics
            model_config.epochs_trained = len(history.history["loss"])
            model_config.val_accuracy = float(max(history.history.get("val_accuracy", [0])))
            model_config.val_loss = float(min(history.history.get("val_loss", [999])))
            model_config.train_accuracy = float(history.history.get("accuracy", [0])[-1])
            model_config.train_loss = float(history.history["loss"][-1])
            
            log.info(f"Model {model_name} - Val Acc: {model_config.val_accuracy:.4f}, "
                    f"Val Loss: {model_config.val_loss:.4f}")
            
            # Clear memory
            tf.keras.backend.clear_session()
            del model
            del trained_model
            
        except Exception as e:
            log.error(f"Error evaluating model {model_name}: {e}")
            # Set poor metrics on failure
            model_config.val_accuracy = 0.0
            model_config.val_loss = 999.0
            model_config.train_accuracy = 0.0
            model_config.train_loss = 999.0
        
        return model_config


# ============================================================================
# RANDOM/GRID SEARCH
# ============================================================================

class RandomSearch:
    """Implements random search over hyperparameter space."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.evaluator = ModelEvaluator(config)
    
    def run(self) -> List[ModelConfig]:
        """
        Run random search to explore hyperparameter space.
        
        Returns:
            List of evaluated model configurations
        """
        log.info("="*80)
        log.info("STARTING RANDOM SEARCH PHASE")
        log.info("="*80)
        log.info(f"Iterations: {self.config.random_search_iterations}")
        log.info(f"Partial training epochs: {self.config.partial_training_epochs}")
        
        results = []
        
        for i in range(self.config.random_search_iterations):
            log.info(f"\n--- Random Search Iteration {i+1}/{self.config.random_search_iterations} ---")
            
            # Sample random architecture
            arch_dict = HyperparameterSpace.sample_architecture()
            
            model_config = ModelConfig(
                conv_layers=arch_dict['conv_layers'],
                conv_activations=arch_dict['conv_activations'],
                pool_size=arch_dict['pool_size'],
                dense_layers=arch_dict['dense_layers'],
                dense_activations=arch_dict['dense_activations'],
                dropout_rates=arch_dict['dropout_rates'],
                learning_rate=arch_dict['learning_rate'],
                generation=0,
                model_id=f"random_search_{i:03d}"
            )
            
            # Evaluate
            model_config = self.evaluator.evaluate_model(
                model_config,
                epochs=self.config.partial_training_epochs,
                model_name=model_config.model_id,
                save_model=False
            )
            
            results.append(model_config)
        
        # Sort by validation accuracy
        results.sort(key=lambda x: x.val_accuracy, reverse=True)
        
        log.info("\n" + "="*80)
        log.info("RANDOM SEARCH COMPLETED")
        log.info("="*80)
        log.info(f"Best validation accuracy: {results[0].val_accuracy:.4f}")
        log.info(f"Best architecture: {results[0].get_architecture_string()}")
        
        return results


# ============================================================================
# NEUROEVOLUTION
# ============================================================================

class NeuroEvolution:
    """Implements neuroevolution with mutation-based architecture search."""
    
    def __init__(self, config: OptimizationConfig, baseline_config: ModelConfig):
        self.config = config
        self.baseline_config = baseline_config
        self.evaluator = ModelEvaluator(config)
        self.generation_history = []
    
    def initialize_population(self) -> List[ModelConfig]:
        """Create initial population from baseline with mutations."""
        log.info(f"Initializing population of {self.config.population_size} models")
        
        population = []
        
        # Add baseline
        baseline = copy.deepcopy(self.baseline_config)
        baseline.generation = 1
        baseline.model_id = "gen1_model_000_baseline"
        population.append(baseline)
        
        # Generate mutated variants
        for i in range(1, self.config.population_size):
            mutated = MutationOperators.mutate(
                copy.deepcopy(self.baseline_config),
                self.config.mutation_rate
            )
            mutated.generation = 1
            mutated.parent_id = "baseline"
            mutated.model_id = f"gen1_model_{i:03d}"
            population.append(mutated)
        
        return population
    
    def select_survivors(self, population: List[ModelConfig]) -> List[ModelConfig]:
        """Select top performers for next generation."""
        # Sort by validation accuracy
        population.sort(key=lambda x: x.val_accuracy, reverse=True)
        
        # Select top models
        n_survivors = max(1, int(len(population) * self.config.selection_ratio))
        survivors = population[:n_survivors]
        
        log.info(f"Selected {n_survivors} survivors for next generation")
        log.info(f"Best val accuracy: {survivors[0].val_accuracy:.4f}")
        log.info(f"Worst survivor val accuracy: {survivors[-1].val_accuracy:.4f}")
        
        return survivors
    
    def create_offspring(
        self,
        parents: List[ModelConfig],
        generation: int
    ) -> List[ModelConfig]:
        """Create new generation through mutation."""
        offspring = []
        
        # Keep best parent unchanged (elitism)
        best_parent = copy.deepcopy(parents[0])
        best_parent.generation = generation
        best_parent.model_id = f"gen{generation}_model_000_elite"
        offspring.append(best_parent)
        
        # Create mutated offspring
        offspring_count = 1
        while len(offspring) < self.config.population_size:
            # Select random parent
            parent = random.choice(parents)
            
            # Mutate
            child = MutationOperators.mutate(
                copy.deepcopy(parent),
                self.config.mutation_rate
            )
            child.generation = generation
            child.parent_id = parent.model_id
            child.model_id = f"gen{generation}_model_{offspring_count:03d}"
            
            offspring.append(child)
            offspring_count += 1
        
        return offspring
    
    def run_generation(
        self,
        population: List[ModelConfig],
        generation: int
    ) -> List[ModelConfig]:
        """Run one generation: evaluate all models."""
        log.info(f"\n{'='*80}")
        log.info(f"GENERATION {generation}")
        log.info(f"{'='*80}")
        log.info(f"Population size: {len(population)}")
        
        results = []
        
        for i, model_config in enumerate(population):
            log.info(f"\n--- Model {i+1}/{len(population)} ---")
            
            # Evaluate
            evaluated = self.evaluator.evaluate_model(
                model_config,
                epochs=self.config.partial_training_epochs,
                model_name=model_config.model_id,
                save_model=False
            )
            
            results.append(evaluated)
        
        # Log generation summary
        results.sort(key=lambda x: x.val_accuracy, reverse=True)
        avg_val_acc = np.mean([r.val_accuracy for r in results])
        
        log.info(f"\nGeneration {generation} Summary:")
        log.info(f"  Best val accuracy: {results[0].val_accuracy:.4f}")
        log.info(f"  Average val accuracy: {avg_val_acc:.4f}")
        log.info(f"  Best architecture: {results[0].get_architecture_string()}")
        
        self.generation_history.append({
            'generation': generation,
            'best_val_acc': results[0].val_accuracy,
            'avg_val_acc': avg_val_acc,
            'best_model_id': results[0].model_id
        })
        
        return results
    
    def run(self) -> ModelConfig:
        """
        Run complete neuroevolution process.
        
        Returns:
            Best model configuration found
        """
        log.info("="*80)
        log.info("STARTING NEUROEVOLUTION PHASE")
        log.info("="*80)
        log.info(f"Generations: {self.config.num_generations}")
        log.info(f"Population size: {self.config.population_size}")
        log.info(f"Mutation rate: {self.config.mutation_rate}")
        
        # Initialize population
        population = self.initialize_population()
        
        all_models = []
        
        # Evolution loop
        for gen in range(1, self.config.num_generations + 1):
            # Evaluate population
            evaluated_population = self.run_generation(population, gen)
            all_models.extend(evaluated_population)
            
            # Select survivors
            survivors = self.select_survivors(evaluated_population)
            
            # Create next generation (unless last generation)
            if gen < self.config.num_generations:
                population = self.create_offspring(survivors, gen + 1)
        
        # Find best overall model
        all_models.sort(key=lambda x: x.val_accuracy, reverse=True)
        best_model = all_models[0]
        
        log.info("\n" + "="*80)
        log.info("NEUROEVOLUTION COMPLETED")
        log.info("="*80)
        log.info(f"Best model: {best_model.model_id}")
        log.info(f"Best val accuracy: {best_model.val_accuracy:.4f}")
        log.info(f"From generation: {best_model.generation}")
        
        return best_model


# ============================================================================
# OPTIMIZATION PIPELINE
# ============================================================================

class OptimizationPipeline:
    """Main orchestrator for the optimization pipeline."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.setup_directories()
        self.all_models = []
    
    def setup_directories(self):
        """Create output directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
        log.info(f"Output directory: {self.config.output_dir}")
    
    def create_baseline_config(self) -> ModelConfig:
        """Create baseline model configuration."""
        return ModelConfig(
            conv_layers=[(32, (3, 3)), (64, (3, 3)), (128, (3, 3))],
            conv_activations=['relu', 'relu', 'relu'],
            pool_size=(2, 2),
            dense_layers=[128, 64],
            dense_activations=['relu', 'relu'],
            dropout_rates=[0.2, 0.3, 0.4],
            learning_rate=0.001,
            generation=0,
            model_id="baseline"
        )
    
    def save_all_models_log(self):
        """Save comprehensive log of all evaluated models."""
        if not self.all_models:
            return
        
        records = []
        for model in self.all_models:
            record = model.to_dict()
            # Convert lists to strings for CSV
            record['conv_layers_str'] = str(model.conv_layers)
            record['dense_layers_str'] = str(model.dense_layers)
            record['architecture'] = model.get_architecture_string()
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(self.config.optimization_log_csv, index=False)
        log.info(f"Saved optimization log to {self.config.optimization_log_csv}")
    
    def train_best_model_fully(self, best_config: ModelConfig) -> Tuple[tf.keras.Model, ModelConfig]:
        """Train the best model with full epochs."""
        log.info("="*80)
        log.info("TRAINING BEST MODEL WITH FULL EPOCHS")
        log.info("="*80)
        log.info(f"Model: {best_config.model_id}")
        log.info(f"Architecture: {best_config.get_architecture_string()}")
        
        # Build model
        model_name = f"best_model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model = build_and_compile_model_03(
            model_name=model_name,
            input_shape=ModelDefaults.INPUT_SHAPE,
            output_units=ModelDefaults.OUTPUT_UNITS,
            conv_layers=best_config.conv_layers,
            conv_activations=best_config.conv_activations,
            pool_size=best_config.pool_size,
            dense_layers=best_config.dense_layers,
            dense_activations=best_config.dense_activations,
            dropout_rates=best_config.dropout_rates,
            optimizer=ModelDefaults.OPTIMIZER,
            learning_rate=best_config.learning_rate,
            loss=ModelDefaults.LOSS,
            metrics=ModelDefaults.METRICS,
            save=False
        )
        
        # Train
        trained_model, history = train_model_pipeline_04(
            model=model,
            tfrecord_dir=self.config.tfrecord_dir,
            batch_size=self.config.batch_size,
            epochs=self.config.full_training_epochs,
            learning_rate=best_config.learning_rate,
            early_stopping_patience=self.config.early_stopping_patience,
            save=True,
            model_save_dir=self.config.models_dir,
            model_registry_csv=self.config.optimization_log_csv,
            notes=f"Best model from optimization - {best_config.model_id}",
            cache_dataset=False
        )
        
        # Update config with final metrics
        best_config.epochs_trained = len(history.history["loss"])
        best_config.val_accuracy = float(max(history.history.get("val_accuracy", [0])))
        best_config.val_loss = float(min(history.history.get("val_loss", [999])))
        best_config.train_accuracy = float(history.history.get("accuracy", [0])[-1])
        best_config.train_loss = float(history.history["loss"][-1])
        
        log.info(f"Final training - Val Acc: {best_config.val_accuracy:.4f}")
        
        return trained_model, best_config
    
    def evaluate_on_test(self, model: tf.keras.Model, model_config: ModelConfig):
        """Evaluate final model on test set."""
        log.info("="*80)
        log.info("EVALUATING ON TEST SET")
        log.info("="*80)
        
        test_results = evaluate_model_on_test(
            model=model,
            tfrecord_dir=self.config.tfrecord_dir,
            batch_size=self.config.batch_size
        )
        
        model_config.test_accuracy = test_results['test_accuracy']
        model_config.test_loss = test_results['test_loss']
        
        log.info(f"Test accuracy: {model_config.test_accuracy:.4f}")
        log.info(f"Test loss: {model_config.test_loss:.4f}")
        
        return test_results
    
    def generate_final_report(self, best_config: ModelConfig, test_results: dict):
        """Generate comprehensive final report."""
        report_path = os.path.join(self.config.output_dir, "final_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL OPTIMIZATION PIPELINE - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Population size: {self.config.population_size}\n")
            f.write(f"Generations: {self.config.num_generations}\n")
            f.write(f"Random search iterations: {self.config.random_search_iterations}\n")
            f.write(f"Partial training epochs: {self.config.partial_training_epochs}\n")
            f.write(f"Full training epochs: {self.config.full_training_epochs}\n")
            f.write(f"Batch size: {self.config.batch_size}\n\n")
            
            f.write("BEST MODEL\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model ID: {best_config.model_id}\n")
            f.write(f"Generation: {best_config.generation}\n")
            f.write(f"Parent ID: {best_config.parent_id}\n\n")
            
            f.write("ARCHITECTURE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Architecture string: {best_config.get_architecture_string()}\n\n")
            
            f.write("Convolutional layers:\n")
            for i, (filters, kernel) in enumerate(best_config.conv_layers):
                f.write(f"  Layer {i+1}: {filters} filters, kernel {kernel}, "
                       f"activation '{best_config.conv_activations[i]}', "
                       f"dropout {best_config.dropout_rates[i]}\n")
            
            f.write(f"\nPooling size: {best_config.pool_size}\n\n")
            
            f.write("Dense layers:\n")
            for i, neurons in enumerate(best_config.dense_layers):
                f.write(f"  Layer {i+1}: {neurons} neurons, "
                       f"activation '{best_config.dense_activations[i]}'\n")
            
            f.write(f"\nLearning rate: {best_config.learning_rate}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training accuracy: {best_config.train_accuracy:.4f}\n")
            f.write(f"Training loss: {best_config.train_loss:.4f}\n")
            f.write(f"Validation accuracy: {best_config.val_accuracy:.4f}\n")
            f.write(f"Validation loss: {best_config.val_loss:.4f}\n")
            f.write(f"Test accuracy: {best_config.test_accuracy:.4f}\n")
            f.write(f"Test loss: {best_config.test_loss:.4f}\n")
            f.write(f"Epochs trained: {best_config.epochs_trained}\n\n")
            
            f.write("TOTAL MODELS EVALUATED\n")
            f.write("-" * 40 + "\n")
            f.write(f"Random search: {self.config.random_search_iterations}\n")
            f.write(f"Neuroevolution: {self.config.population_size * self.config.num_generations}\n")
            f.write(f"Total: {len(self.all_models)}\n\n")
            
            f.write("="*80 + "\n")
        
        log.info(f"Final report saved to {report_path}")
    
    def run(self):
        """Run the complete optimization pipeline."""
        start_time = datetime.now()
        
        log.info("\n" + "="*80)
        log.info("STARTING MODEL OPTIMIZATION PIPELINE")
        log.info("="*80)
        log.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Set random seeds
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        tf.random.set_seed(self.config.random_seed)
        
        # Phase 1: Random Search
        random_search = RandomSearch(self.config)
        random_results = random_search.run()
        self.all_models.extend(random_results)
        
        # Use best from random search as baseline for evolution
        best_random = random_results[0]
        
        # Phase 2: NeuroEvolution
        neuroevolution = NeuroEvolution(self.config, best_random)
        best_evolved = neuroevolution.run()
        self.all_models.extend([best_evolved])  # Already included in generation results
        
        # Save all models log
        self.save_all_models_log()
        
        # Phase 3: Full training of best model
        best_model, best_config = self.train_best_model_fully(best_evolved)
        
        # Phase 4: Test evaluation
        test_results = self.evaluate_on_test(best_model, best_config)
        
        # Phase 5: Generate report
        self.generate_final_report(best_config, test_results)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        log.info("\n" + "="*80)
        log.info("OPTIMIZATION PIPELINE COMPLETED")
        log.info("="*80)
        log.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Total duration: {duration}")
        log.info(f"Best model test accuracy: {best_config.test_accuracy:.4f}")
        log.info(f"Output directory: {self.config.output_dir}")
        log.info("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for the optimization pipeline."""
    
    # Create configuration
    config = OptimizationConfig(
        # Data paths
        tfrecord_dir="data/tfrecords",  # Adjust to your data path
        
        # Resource constraints
        batch_size=32,
        partial_training_epochs=10,
        full_training_epochs=50,
        early_stopping_patience=5,
        
        # Random search
        random_search_iterations=10,
        
        # Neuroevolution
        population_size=8,
        num_generations=5,
        mutation_rate=0.3,
        selection_ratio=0.5,
        
        # Output
        output_dir="optimization_results",
        
        # Seed
        random_seed=42
    )
    
    # Create and run pipeline
    pipeline = OptimizationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()