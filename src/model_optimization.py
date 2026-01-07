"""
Automatic Model Optimization Pipeline for Music Genre Classification
=====================================================================

This script implements a hybrid optimization strategy combining:
1. Random/Grid Search for baseline exploration
2. NeuroEvolution for iterative architecture refinement
3. Comprehensive evaluation and logging

VERSION: 2.0 - Enhanced with complete model saving and registry tracking
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
from sklearn.metrics import classification_report, confusion_matrix
import copy

# Import project modules
from src.cste import *
from src.logger import get_logger
from src.data_utils import compute_input_shape
from model_generator import build_and_compile_model_03
from model_training import train_model_pipeline_04, evaluate_model_on_test

log = get_logger("model_optimization")


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline."""

    # Data paths
    tfrecord_dir: str = TFRECORD_OUTPUT_DIR

    # Input shape will be computed in __post_init__
    input_shape: Tuple[int, int, int] = None

    # Resource constraints
    batch_size: int = 32
    partial_training_epochs: int = 10
    full_training_epochs: int = 50
    early_stopping_patience: int = 5

    # Random/Grid Search
    random_search_iterations: int = 10

    # NeuroEvolution
    population_size: int = 8
    num_generations: int = 5
    mutation_rate: float = 0.3
    selection_ratio: float = 0.5

    # Output paths
    output_dir: str = "optimization_results"
    models_dir: str = "optimization_results/models"
    logs_dir: str = "optimization_results/logs"

    # Random seed
    random_seed: int = 42

    def __post_init__(self):
        """Compute input_shape after instance creation."""
        if self.input_shape is None:
            normalization_file = os.path.join(
                self.tfrecord_dir, "normalization_stats.json"
            )

            if not os.path.exists(normalization_file):
                log.warning(f"Normalization file not found: {normalization_file}")
                log.warning(f"Using default INPUT_SHAPE from ModelDefaults")
                self.input_shape = ModelDefaults.INPUT_SHAPE
            else:
                try:
                    self.input_shape = compute_input_shape(
                        normalization_file=normalization_file,
                        audio_duration=SEGMENT_DURATION,
                        channels=NUM_CHANNELS,
                        verbose=False,
                    )
                    log.info(f"Computed INPUT_SHAPE: {self.input_shape}")
                except Exception as e:
                    log.error(f"Error computing input_shape: {e}")
                    log.info(f"Using default INPUT_SHAPE from ModelDefaults")
                    self.input_shape = ModelDefaults.INPUT_SHAPE


# ============================================================================
# HYPERPARAMETER SPACE DEFINITION
# ============================================================================


class HyperparameterSpace:
    """Defines the search space for model hyperparameters."""

    # Convolutional layer options
    CONV_FILTERS = [16, 32, 64, 128]
    KERNEL_SIZES = [(3, 3), (5, 5), (7, 7)]
    POOL_SIZES = [(2, 2), (3, 3)]
    CONV_ACTIVATIONS = ["relu", "elu", "selu"]

    # Dense layer options
    DENSE_NEURONS = [64, 128, 256, 512]
    DENSE_ACTIVATIONS = ["relu", "elu", "selu"]

    # Regularization
    DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Training hyperparameters
    LEARNING_RATES = [0.0005, 0.001, 0.005, 0.01]

    # Architecture depth
    MIN_CONV_LAYERS = 2
    MAX_CONV_LAYERS = 4
    MIN_DENSE_LAYERS = 1
    MAX_DENSE_LAYERS = 3

    @staticmethod
    def sample_architecture() -> Dict:
        """Sample a random architecture configuration."""
        num_conv_layers = random.randint(
            HyperparameterSpace.MIN_CONV_LAYERS, HyperparameterSpace.MAX_CONV_LAYERS
        )
        num_dense_layers = random.randint(
            HyperparameterSpace.MIN_DENSE_LAYERS, HyperparameterSpace.MAX_DENSE_LAYERS
        )

        conv_layers = [
            (
                random.choice(HyperparameterSpace.CONV_FILTERS),
                random.choice(HyperparameterSpace.KERNEL_SIZES),
            )
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
            "conv_layers": conv_layers,
            "conv_activations": conv_activations,
            "pool_size": random.choice(HyperparameterSpace.POOL_SIZES),
            "dense_layers": dense_layers,
            "dense_activations": dense_activations,
            "dropout_rates": dropout_rates,
            "learning_rate": random.choice(HyperparameterSpace.LEARNING_RATES),
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

    # ✨ MODIFICATION 1: Nouveaux champs pour le registry
    timestamp: Optional[str] = None
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    dataset_csv_path: Optional[str] = None
    batch_size: Optional[int] = None
    framework: str = "tensorflow"
    notes: Optional[str] = None

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
            factor = random.uniform(0.5, 2.0)
            config.learning_rate = np.clip(config.learning_rate * factor, 0.00001, 0.01)
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
            log.debug(
                f"Mutated dense neurons at layer {idx} to {config.dense_layers[idx]}"
            )
        return config

    @staticmethod
    def add_conv_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Add a convolutional layer (large change, low probability)."""
        if (
            random.random() < mutation_rate * 0.3
            and len(config.conv_layers) < HyperparameterSpace.MAX_CONV_LAYERS
        ):

            new_layer = (
                random.choice(HyperparameterSpace.CONV_FILTERS),
                random.choice(HyperparameterSpace.KERNEL_SIZES),
            )
            config.conv_layers.append(new_layer)
            config.conv_activations.append(
                random.choice(HyperparameterSpace.CONV_ACTIVATIONS)
            )
            config.dropout_rates.append(
                random.choice(HyperparameterSpace.DROPOUT_RATES)
            )
            log.debug(f"Added conv layer: {new_layer}")
        return config

    @staticmethod
    def remove_conv_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Remove a convolutional layer (large change, low probability)."""
        if (
            random.random() < mutation_rate * 0.2
            and len(config.conv_layers) > HyperparameterSpace.MIN_CONV_LAYERS
        ):

            idx = random.randint(0, len(config.conv_layers) - 1)
            config.conv_layers.pop(idx)
            config.conv_activations.pop(idx)
            config.dropout_rates.pop(idx)
            log.debug(f"Removed conv layer at index {idx}")
        return config

    @staticmethod
    def add_dense_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Add a dense layer (large change, low probability)."""
        if (
            random.random() < mutation_rate * 0.3
            and len(config.dense_layers) < HyperparameterSpace.MAX_DENSE_LAYERS
        ):

            new_neurons = random.choice(HyperparameterSpace.DENSE_NEURONS)
            config.dense_layers.append(new_neurons)
            config.dense_activations.append(
                random.choice(HyperparameterSpace.DENSE_ACTIVATIONS)
            )
            log.debug(f"Added dense layer: {new_neurons} neurons")
        return config

    @staticmethod
    def remove_dense_layer(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Remove a dense layer (large change, low probability)."""
        if (
            random.random() < mutation_rate * 0.2
            and len(config.dense_layers) > HyperparameterSpace.MIN_DENSE_LAYERS
        ):

            idx = random.randint(0, len(config.dense_layers) - 1)
            config.dense_layers.pop(idx)
            config.dense_activations.pop(idx)
            log.debug(f"Removed dense layer at index {idx}")
        return config

    @staticmethod
    def mutate(config: ModelConfig, mutation_rate: float) -> ModelConfig:
        """Apply all mutation operators to a config."""
        mutated = copy.deepcopy(config)

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
    """Handles model training and evaluation with detailed per-class metrics."""

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def evaluate_model(
        self,
        model_config: ModelConfig,
        epochs: int,
        model_name: str,
        save_model: bool = False,
        model_save_path: Optional[str] = None,  # ✨ MODIFICATION 4: Nouveau paramètre
        detailed_eval: bool = False,
    ) -> ModelConfig:
        """
        Train and evaluate a model configuration.

        Args:
            model_config: Model configuration to evaluate
            epochs: Number of training epochs
            model_name: Unique name for this model
            save_model: Whether to save the trained model
            model_save_path: Explicit path where to save the model (.keras)
            detailed_eval: Whether to compute detailed per-class metrics

        Returns:
            Updated model_config with performance metrics
        """
        log.info(f"Evaluating model: {model_name}")
        log.info(f"Architecture: {model_config.get_architecture_string()}")

        try:
            # Build model
            model = build_and_compile_model_03(
                model_name=model_name,
                input_shape=self.config.input_shape,
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
                save=False,
            )

            trained_model, history = train_model_pipeline_04(
                model=model,
                tfrecord_dir=self.config.tfrecord_dir,
                batch_size=self.config.batch_size,
                epochs=epochs,
                learning_rate=model_config.learning_rate,
                early_stopping_patience=self.config.early_stopping_patience,
                save=False,  # On sauvegarde manuellement
                model_registry_csv=None,
                notes=f"Generation {model_config.generation}",
                cache_dataset=False,
                shuffle_buffer_size=1000,
            )

            # ✨ MODIFICATION 4: Sauvegarder manuellement si demandé
            if save_model and model_save_path:
                trained_model.save(model_save_path)
                log.info(f"Model saved to: {model_save_path}")

                # Mettre à jour model_config
                model_config.model_path = model_save_path
                model_config.model_name = os.path.splitext(
                    os.path.basename(model_save_path)
                )[0]

            # Extract basic metrics
            model_config.epochs_trained = len(history.history["loss"])
            model_config.val_accuracy = float(
                max(history.history.get("val_accuracy", [0]))
            )
            model_config.val_loss = float(min(history.history.get("val_loss", [999])))
            model_config.train_accuracy = float(
                history.history.get("accuracy", [0])[-1]
            )
            model_config.train_loss = float(history.history["loss"][-1])

            # Detailed evaluation if requested (for final model)
            if detailed_eval:
                detailed_metrics = self.compute_detailed_metrics(
                    trained_model, model_name
                )
                if not hasattr(model_config, "detailed_metrics"):
                    model_config.detailed_metrics = {}
                model_config.detailed_metrics = detailed_metrics

            log.info(
                f"Model {model_name} - Val Acc: {model_config.val_accuracy:.4f}, "
                f"Val Loss: {model_config.val_loss:.4f}"
            )

            # Clear memory
            tf.keras.backend.clear_session()
            del model
            del trained_model

        except Exception as e:
            log.error(f"Error evaluating model {model_name}: {e}")
            model_config.val_accuracy = 0.0
            model_config.val_loss = 999.0
            model_config.train_accuracy = 0.0
            model_config.train_loss = 999.0

        return model_config

    def compute_detailed_metrics(self, model: tf.keras.Model, model_name: str) -> dict:
        """Compute detailed per-class metrics on validation set."""
        log.info(f"Computing detailed metrics for {model_name}...")

        try:
            val_dataset = self._load_validation_dataset()

            y_true = []
            y_pred = []

            for batch_x, batch_y in val_dataset:
                predictions = model.predict(batch_x, verbose=0)
                y_pred.extend(np.argmax(predictions, axis=1))
                y_true.extend(np.argmax(batch_y.numpy(), axis=1))

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            cm = confusion_matrix(y_true, y_pred)
            class_names = self._get_class_names()

            report = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )

            per_class_accuracy = {}
            for i, class_name in enumerate(class_names):
                mask = y_true == i
                if mask.sum() > 0:
                    per_class_accuracy[class_name] = (y_pred[mask] == i).mean()
                else:
                    per_class_accuracy[class_name] = 0.0

            detailed_metrics = {
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "per_class_accuracy": per_class_accuracy,
                "per_class_precision": {
                    cls: report[cls]["precision"] for cls in class_names
                },
                "per_class_recall": {cls: report[cls]["recall"] for cls in class_names},
                "per_class_f1": {cls: report[cls]["f1-score"] for cls in class_names},
                "macro_avg_precision": report["macro avg"]["precision"],
                "macro_avg_recall": report["macro avg"]["recall"],
                "macro_avg_f1": report["macro avg"]["f1-score"],
                "weighted_avg_precision": report["weighted avg"]["precision"],
                "weighted_avg_recall": report["weighted avg"]["recall"],
                "weighted_avg_f1": report["weighted avg"]["f1-score"],
            }

            log.info(f"Detailed metrics computed successfully")
            log.info(f"  Macro avg F1: {detailed_metrics['macro_avg_f1']:.4f}")
            log.info(f"  Weighted avg F1: {detailed_metrics['weighted_avg_f1']:.4f}")

            return detailed_metrics

        except Exception as e:
            log.error(f"Error computing detailed metrics: {e}")
            return {}

    def _load_validation_dataset(self):
        """Load validation dataset from TFRecords."""
        val_tfrecord = os.path.join(self.config.tfrecord_dir, "val.tfrecord")

        dataset = tf.data.TFRecordDataset(val_tfrecord)
        dataset = dataset.map(self._parse_tfrecord)
        dataset = dataset.batch(self.config.batch_size)

        return dataset

    def _parse_tfrecord(self, example_proto):
        """Parse TFRecord example."""
        feature_description = {
            "audio": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        parsed = tf.io.parse_single_example(example_proto, feature_description)

        audio = tf.io.parse_tensor(parsed["audio"], out_type=tf.float32)
        audio = tf.reshape(audio, self.config.input_shape)

        label = tf.one_hot(parsed["label"], depth=ModelDefaults.OUTPUT_UNITS)

        return audio, label

    def _get_class_names(self) -> list:
        """Get list of class names."""
        if hasattr(ModelDefaults, "CLASS_NAMES"):
            return ModelDefaults.CLASS_NAMES

        return [
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        ]

    def _save_detailed_metrics_csv(self, metrics: dict, output_dir: str):
        """Save per-class metrics to CSV for easy analysis."""
        csv_path = os.path.join(output_dir, "per_class_metrics.csv")

        records = []
        for class_name in sorted(metrics["per_class_accuracy"].keys()):
            records.append(
                {
                    "class": class_name,
                    "accuracy": metrics["per_class_accuracy"][class_name],
                    "precision": metrics["per_class_precision"][class_name],
                    "recall": metrics["per_class_recall"][class_name],
                    "f1_score": metrics["per_class_f1"][class_name],
                }
            )

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        log.info(f"Per-class metrics saved to {csv_path}")


# ============================================================================
# RANDOM/GRID SEARCH
# ============================================================================


class RandomSearch:
    """Implements random search over hyperparameter space."""

    def __init__(
        self, config: OptimizationConfig, pipeline: "OptimizationPipeline"
    ):  # ✨ MODIFICATION 8
        self.config = config
        self.pipeline = pipeline
        self.evaluator = ModelEvaluator(config)

    def run(self) -> List[ModelConfig]:
        """Run random search to explore hyperparameter space."""
        log.info("=" * 80)
        log.info("STARTING RANDOM SEARCH PHASE")
        log.info("=" * 80)
        log.info(f"Iterations: {self.config.random_search_iterations}")
        log.info(f"Partial training epochs: {self.config.partial_training_epochs}")

        results = []

        for i in range(self.config.random_search_iterations):
            log.info(
                f"\n--- Random Search Iteration {i+1}/{self.config.random_search_iterations} ---"
            )

            arch_dict = HyperparameterSpace.sample_architecture()

            model_config = ModelConfig(
                conv_layers=arch_dict["conv_layers"],
                conv_activations=arch_dict["conv_activations"],
                pool_size=arch_dict["pool_size"],
                dense_layers=arch_dict["dense_layers"],
                dense_activations=arch_dict["dense_activations"],
                dropout_rates=arch_dict["dropout_rates"],
                learning_rate=arch_dict["learning_rate"],
                generation=0,
                model_id=f"rd_search_{i:03d}",  # ✨ Convention de nommage
            )

            # ✨ MODIFICATION 3: Définir le chemin de sauvegarde
            model_save_path = os.path.join(
                self.pipeline.random_search_models_dir, f"{model_config.model_id}.keras"
            )

            # Evaluate et sauvegarder
            model_config = self.evaluator.evaluate_model(
                model_config,
                epochs=self.config.partial_training_epochs,
                model_name=model_config.model_id,
                save_model=True,  # ✨ Sauvegarder
                model_save_path=model_save_path,
            )

            # ✨ MODIFICATION 3: Enregistrer les métadonnées
            model_config.model_path = model_save_path
            model_config.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_config.batch_size = self.config.batch_size
            model_config.dataset_csv_path = self.config.tfrecord_dir
            model_config.notes = f"Random Search iteration {i+1}"

            results.append(model_config)

            # ✨ MODIFICATION 8: Enregistrer dans le CSV
            self.pipeline.append_to_registry(model_config)

        results.sort(key=lambda x: x.val_accuracy, reverse=True)

        log.info("\n" + "=" * 80)
        log.info("RANDOM SEARCH COMPLETED")
        log.info("=" * 80)
        log.info(f"Best validation accuracy: {results[0].val_accuracy:.4f}")
        log.info(f"Best architecture: {results[0].get_architecture_string()}")

        return results


# ============================================================================
# NEUROEVOLUTION
# ============================================================================


class NeuroEvolution:
    """Implements neuroevolution with mutation-based architecture search."""

    def __init__(
        self,
        config: OptimizationConfig,
        baseline_config: ModelConfig,
        pipeline: "OptimizationPipeline",
    ):  # ✨ MODIFICATION 8
        self.config = config
        self.baseline_config = baseline_config
        self.pipeline = pipeline
        self.evaluator = ModelEvaluator(config)
        self.generation_history = []

    def initialize_population(self) -> List[ModelConfig]:
        """Create initial population from baseline with mutations."""
        log.info(f"Initializing population of {self.config.population_size} models")

        population = []

        baseline = copy.deepcopy(self.baseline_config)
        baseline.generation = 1
        baseline.model_id = "gen1_model_000_baseline"
        population.append(baseline)

        for i in range(1, self.config.population_size):
            mutated = MutationOperators.mutate(
                copy.deepcopy(self.baseline_config), self.config.mutation_rate
            )
            mutated.generation = 1
            mutated.parent_id = "baseline"
            mutated.model_id = f"gen1_model_{i:03d}"
            population.append(mutated)

        return population

    def select_survivors(self, population: List[ModelConfig]) -> List[ModelConfig]:
        """Select top performers for next generation."""
        population.sort(key=lambda x: x.val_accuracy, reverse=True)

        n_survivors = max(1, int(len(population) * self.config.selection_ratio))
        survivors = population[:n_survivors]

        log.info(f"Selected {n_survivors} survivors for next generation")
        log.info(f"Best val accuracy: {survivors[0].val_accuracy:.4f}")
        log.info(f"Worst survivor val accuracy: {survivors[-1].val_accuracy:.4f}")

        return survivors

    def create_offspring(
        self, parents: List[ModelConfig], generation: int
    ) -> List[ModelConfig]:
        """Create new generation through mutation."""
        offspring = []

        best_parent = copy.deepcopy(parents[0])
        best_parent.generation = generation
        best_parent.model_id = f"gen{generation}_model_000_elite"
        offspring.append(best_parent)

        offspring_count = 1
        while len(offspring) < self.config.population_size:
            parent = random.choice(parents)

            child = MutationOperators.mutate(
                copy.deepcopy(parent), self.config.mutation_rate
            )
            child.generation = generation
            child.parent_id = parent.model_id
            child.model_id = f"gen{generation}_model_{offspring_count:03d}"

            offspring.append(child)
            offspring_count += 1

        return offspring

    def run_generation(
        self, population: List[ModelConfig], generation: int
    ) -> List[ModelConfig]:
        """Run one generation: evaluate all models."""
        log.info(f"\n{'='*80}")
        log.info(f"GENERATION {generation}")
        log.info(f"{'='*80}")
        log.info(f"Population size: {len(population)}")

        results = []

        for i, model_config in enumerate(population):
            log.info(f"\n--- Model {i+1}/{len(population)} ---")

            # ✨ MODIFICATION 5: Définir le chemin de sauvegarde
            model_save_path = os.path.join(
                self.pipeline.neuroevolution_models_dir,
                f"{model_config.model_id}.keras",
            )

            # Evaluate et sauvegarder
            evaluated = self.evaluator.evaluate_model(
                model_config,
                epochs=self.config.partial_training_epochs,
                model_name=model_config.model_id,
                save_model=True,  # ✨ Sauvegarder
                model_save_path=model_save_path,
            )

            # ✨ MODIFICATION 5: Enregistrer les métadonnées
            evaluated.model_path = model_save_path
            evaluated.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            evaluated.batch_size = self.config.batch_size
            evaluated.dataset_csv_path = self.config.tfrecord_dir
            evaluated.notes = f"Generation {generation}, model {i+1}/{len(population)}"

            results.append(evaluated)

            # ✨ MODIFICATION 8: Enregistrer dans le CSV
            self.pipeline.append_to_registry(evaluated)

        results.sort(key=lambda x: x.val_accuracy, reverse=True)
        avg_val_acc = np.mean([r.val_accuracy for r in results])

        log.info(f"\nGeneration {generation} Summary:")
        log.info(f"  Best val accuracy: {results[0].val_accuracy:.4f}")
        log.info(f"  Average val accuracy: {avg_val_acc:.4f}")
        log.info(f"  Best architecture: {results[0].get_architecture_string()}")

        self.generation_history.append(
            {
                "generation": generation,
                "best_val_acc": results[0].val_accuracy,
                "avg_val_acc": avg_val_acc,
                "best_model_id": results[0].model_id,
            }
        )

        return results

    def run(self) -> ModelConfig:
        """Run complete neuroevolution process."""
        log.info("=" * 80)
        log.info("STARTING NEUROEVOLUTION PHASE")
        log.info("=" * 80)
        log.info(f"Generations: {self.config.num_generations}")
        log.info(f"Population size: {self.config.population_size}")
        log.info(f"Mutation rate: {self.config.mutation_rate}")

        population = self.initialize_population()

        all_models = []

        for gen in range(1, self.config.num_generations + 1):
            evaluated_population = self.run_generation(population, gen)
            all_models.extend(evaluated_population)

            survivors = self.select_survivors(evaluated_population)

            if gen < self.config.num_generations:
                population = self.create_offspring(survivors, gen + 1)

        all_models.sort(key=lambda x: x.val_accuracy, reverse=True)
        best_model = all_models[0]

        log.info("\n" + "=" * 80)
        log.info("NEUROEVOLUTION COMPLETED")
        log.info("=" * 80)
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
        # ✨ MODIFICATION 2: Structure détaillée
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.random_search_models_dir = os.path.join(
            self.config.models_dir, "random_search_models"
        )
        self.neuroevolution_models_dir = os.path.join(
            self.config.models_dir, "neuro_evolution_models"
        )
        self.exec_report_dir = os.path.join(self.config.output_dir, "exec_report")

        os.makedirs(self.random_search_models_dir, exist_ok=True)
        os.makedirs(self.neuroevolution_models_dir, exist_ok=True)
        os.makedirs(self.exec_report_dir, exist_ok=True)

        log.info(f"Output directory: {self.config.output_dir}")
        log.info(f"Random Search models: {self.random_search_models_dir}")
        log.info(f"Neuroevolution models: {self.neuroevolution_models_dir}")

    def append_to_registry(self, model_config: ModelConfig):
        """
        ✨ MODIFICATION 7: Ajouter un modèle au CSV de manière incrémentale.
        Évite les doublons et garantit la cohérence.
        """
        registry_path = os.path.join(self.config.output_dir, "models_perf.csv")

        record = {
            "model_id": model_config.model_id,
            "generation": model_config.generation,
            "parent_id": model_config.parent_id if model_config.parent_id else "",
            "val_accuracy": model_config.val_accuracy,
            "val_loss": model_config.val_loss,
            "train_accuracy": model_config.train_accuracy,
            "train_loss": model_config.train_loss,
            "test_accuracy": (
                model_config.test_accuracy if model_config.test_accuracy else ""
            ),
            "test_loss": model_config.test_loss if model_config.test_loss else "",
            "epochs_trained": model_config.epochs_trained,
            "model_name": model_config.model_name if model_config.model_name else "",
            "model_path": model_config.model_path if model_config.model_path else "",
            "timestamp": (
                model_config.timestamp
                if model_config.timestamp
                else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "dataset_csv_path": (
                model_config.dataset_csv_path
                if model_config.dataset_csv_path
                else self.config.tfrecord_dir
            ),
            "batch_size": (
                model_config.batch_size
                if model_config.batch_size
                else self.config.batch_size
            ),
            "framework": model_config.framework,
            "notes": model_config.notes if model_config.notes else "",
            "learning_rate": model_config.learning_rate,
            "architecture": model_config.get_architecture_string(),
            "conv_layers": str(model_config.conv_layers),
            "conv_activations": str(model_config.conv_activations),
            "pool_size": str(model_config.pool_size),
            "dense_layers": str(model_config.dense_layers),
            "dense_activations": str(model_config.dense_activations),
            "dropout_rates": str(model_config.dropout_rates),
        }

        df_new = pd.DataFrame([record])

        if os.path.exists(registry_path):
            df_existing = pd.read_csv(registry_path)
            if model_config.model_id in df_existing["model_id"].values:
                log.warning(
                    f"Model {model_config.model_id} already in registry, skipping"
                )
                return

            df_new.to_csv(registry_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(registry_path, mode="w", header=True, index=False)

        log.info(f"Registered model {model_config.model_id} in {registry_path}")

    def create_baseline_config(self) -> ModelConfig:
        """Create baseline model configuration."""
        return ModelConfig(
            conv_layers=[(32, (3, 3)), (64, (3, 3)), (128, (3, 3))],
            conv_activations=["relu", "relu", "relu"],
            pool_size=(2, 2),
            dense_layers=[128, 64],
            dense_activations=["relu", "relu"],
            dropout_rates=[0.2, 0.3, 0.4],
            learning_rate=0.001,
            generation=0,
            model_id="baseline",
        )

    def train_best_model_fully(
        self, best_config: ModelConfig
    ) -> Tuple[tf.keras.Model, ModelConfig]:
        """Train the best model with full epochs."""
        log.info("=" * 80)
        log.info("TRAINING BEST MODEL WITH FULL EPOCHS")
        log.info("=" * 80)
        log.info(f"Model: {best_config.model_id}")
        log.info(f"Architecture: {best_config.get_architecture_string()}")

        # ✨ MODIFICATION 6: Définir le chemin de sauvegarde du modèle final
        model_name = "final_model"
        final_model_path = os.path.join(self.config.models_dir, f"{model_name}.keras")

        model = build_and_compile_model_03(
            model_name=model_name,
            input_shape=self.config.input_shape,
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
            save=False,
        )

        # Train
        trained_model, history = train_model_pipeline_04(
            model=model,
            tfrecord_dir=self.config.tfrecord_dir,
            batch_size=self.config.batch_size,
            epochs=self.config.full_training_epochs,
            learning_rate=best_config.learning_rate,
            early_stopping_patience=self.config.early_stopping_patience,
            save=False,  # ✨ On sauvegarde manuellement après
            cache_dataset=False,
        )

        # ✨ MODIFICATION 6: Sauvegarder manuellement avec le bon nom
        trained_model.save(final_model_path)
        log.info(f"Final model saved to: {final_model_path}")

        # Update config with final metrics
        best_config.epochs_trained = len(history.history["loss"])
        best_config.val_accuracy = float(max(history.history.get("val_accuracy", [0])))
        best_config.val_loss = float(min(history.history.get("val_loss", [999])))
        best_config.train_accuracy = float(history.history.get("accuracy", [0])[-1])
        best_config.train_loss = float(history.history["loss"][-1])

        # ✨ MODIFICATION 6: Enregistrer les métadonnées du modèle final
        best_config.model_path = final_model_path
        best_config.model_name = model_name
        best_config.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_config.batch_size = self.config.batch_size
        best_config.dataset_csv_path = self.config.tfrecord_dir
        best_config.notes = f"Final model - fully trained from {best_config.model_id}"

        log.info(f"Final training - Val Acc: {best_config.val_accuracy:.4f}")

        return trained_model, best_config

    def evaluate_on_test(self, model: tf.keras.Model, model_config: ModelConfig):
        """Evaluate final model on test set."""
        log.info("=" * 80)
        log.info("EVALUATING ON TEST SET")
        log.info("=" * 80)

        test_results = evaluate_model_on_test(
            model=model,
            tfrecord_dir=self.config.tfrecord_dir,
            batch_size=self.config.batch_size,
        )

        model_config.test_accuracy = test_results["test_accuracy"]
        model_config.test_loss = test_results["test_loss"]

        log.info(f"Test accuracy: {model_config.test_accuracy:.4f}")
        log.info(f"Test loss: {model_config.test_loss:.4f}")

        return test_results

    def generate_final_report(self, best_config: ModelConfig, test_results: dict):
        """
        Generate comprehensive final report.
        ✨ MODIFICATION 9: Déplacer vers exec_report/
        """
        report_path = os.path.join(self.exec_report_dir, "optimization_summary.txt")

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL OPTIMIZATION PIPELINE - FINAL REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Population size: {self.config.population_size}\n")
            f.write(f"Generations: {self.config.num_generations}\n")
            f.write(
                f"Random search iterations: {self.config.random_search_iterations}\n"
            )
            f.write(f"Partial training epochs: {self.config.partial_training_epochs}\n")
            f.write(f"Full training epochs: {self.config.full_training_epochs}\n")
            f.write(f"Batch size: {self.config.batch_size}\n\n")

            f.write("BEST MODEL\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model ID: {best_config.model_id}\n")
            f.write(f"Model Path: {best_config.model_path}\n")
            f.write(f"Generation: {best_config.generation}\n")
            f.write(f"Parent ID: {best_config.parent_id}\n\n")

            f.write("ARCHITECTURE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Architecture string: {best_config.get_architecture_string()}\n\n")

            f.write("Convolutional layers:\n")
            for i, (filters, kernel) in enumerate(best_config.conv_layers):
                f.write(
                    f"  Layer {i+1}: {filters} filters, kernel {kernel}, "
                    f"activation '{best_config.conv_activations[i]}', "
                    f"dropout {best_config.dropout_rates[i]}\n"
                )

            f.write(f"\nPooling size: {best_config.pool_size}\n\n")

            f.write("Dense layers:\n")
            for i, neurons in enumerate(best_config.dense_layers):
                f.write(
                    f"  Layer {i+1}: {neurons} neurons, "
                    f"activation '{best_config.dense_activations[i]}'\n"
                )

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

            # Add detailed metrics if available
            if (
                hasattr(best_config, "detailed_metrics")
                and best_config.detailed_metrics
            ):
                metrics = best_config.detailed_metrics

                f.write("DETAILED PERFORMANCE METRICS\n")
                f.write("-" * 40 + "\n\n")

                f.write("Macro Averages:\n")
                f.write(f"  Precision: {metrics['macro_avg_precision']:.4f}\n")
                f.write(f"  Recall: {metrics['macro_avg_recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['macro_avg_f1']:.4f}\n\n")

                f.write("Weighted Averages:\n")
                f.write(f"  Precision: {metrics['weighted_avg_precision']:.4f}\n")
                f.write(f"  Recall: {metrics['weighted_avg_recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['weighted_avg_f1']:.4f}\n\n")

                f.write("PER-CLASS PERFORMANCE\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
                )
                f.write("-" * 60 + "\n")

                for class_name in sorted(metrics["per_class_accuracy"].keys()):
                    acc = metrics["per_class_accuracy"][class_name]
                    prec = metrics["per_class_precision"][class_name]
                    rec = metrics["per_class_recall"][class_name]
                    f1 = metrics["per_class_f1"][class_name]

                    f.write(
                        f"{class_name:<15} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}\n"
                    )

                f.write("\n")

            f.write("TOTAL MODELS EVALUATED\n")
            f.write("-" * 40 + "\n")
            f.write(f"Random search: {self.config.random_search_iterations}\n")
            f.write(
                f"Neuroevolution: {self.config.population_size * self.config.num_generations}\n"
            )
            f.write(f"Total: {len(self.all_models)}\n\n")

            f.write("=" * 80 + "\n")

        log.info(f"Final report saved to {report_path}")

        # Save detailed metrics if available
        if hasattr(best_config, "detailed_metrics") and best_config.detailed_metrics:
            evaluator = ModelEvaluator(self.config)
            evaluator._save_detailed_metrics_csv(
                best_config.detailed_metrics, self.exec_report_dir
            )

    def run(self):
        """Run the complete optimization pipeline."""
        start_time = datetime.now()

        log.info("\n" + "=" * 80)
        log.info("STARTING MODEL OPTIMIZATION PIPELINE")
        log.info("=" * 80)
        log.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        tf.random.set_seed(self.config.random_seed)

        # Phase 1: Random Search
        # ✨ MODIFICATION 8: Passer self à RandomSearch
        random_search = RandomSearch(self.config, pipeline=self)
        random_results = random_search.run()
        self.all_models.extend(random_results)

        best_random = random_results[0]

        # Phase 2: NeuroEvolution
        # ✨ MODIFICATION 8: Passer self à NeuroEvolution
        neuroevolution = NeuroEvolution(self.config, best_random, pipeline=self)
        best_evolved = neuroevolution.run()
        self.all_models.extend([best_evolved])

        # ✨ MODIFICATION 10: Ne plus appeler save_all_models_log (obsolète)
        # Les modèles sont déjà enregistrés au fur et à mesure via append_to_registry

        # Phase 3: Full training of best model
        best_model, best_config = self.train_best_model_fully(best_evolved)

        # Enregistrer le modèle final dans le registry
        self.append_to_registry(best_config)

        # Phase 4: Test evaluation
        test_results = self.evaluate_on_test(best_model, best_config)

        # Mettre à jour le registry avec les résultats de test
        # (Re-enregistrer avec les nouvelles métriques)
        registry_path = os.path.join(self.config.output_dir, "models_perf.csv")
        if os.path.exists(registry_path):
            df = pd.read_csv(registry_path)
            mask = df["model_id"] == best_config.model_id
            df.loc[mask, "test_accuracy"] = best_config.test_accuracy
            df.loc[mask, "test_loss"] = best_config.test_loss
            df.to_csv(registry_path, index=False)

        # Phase 5: Generate report
        self.generate_final_report(best_config, test_results)

        end_time = datetime.now()
        duration = end_time - start_time

        log.info("\n" + "=" * 80)
        log.info("OPTIMIZATION PIPELINE COMPLETED")
        log.info("=" * 80)
        log.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Total duration: {duration}")
        log.info(f"Best model test accuracy: {best_config.test_accuracy:.4f}")
        log.info(f"Output directory: {self.config.output_dir}")
        log.info(
            f"Models performance CSV: {os.path.join(self.config.output_dir, 'models_perf.csv')}"
        )
        log.info("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main entry point for the optimization pipeline."""

    config = OptimizationConfig(
        tfrecord_dir="data/tfrecords",
        batch_size=32,
        partial_training_epochs=10,
        full_training_epochs=50,
        early_stopping_patience=5,
        random_search_iterations=10,
        population_size=8,
        num_generations=5,
        mutation_rate=0.3,
        selection_ratio=0.5,
        output_dir="optimization_results",
        random_seed=42,
    )

    pipeline = OptimizationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
