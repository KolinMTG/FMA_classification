# Model Optimization Pipeline for Music Genre Classification

## Overview

This pipeline implements an advanced model optimization strategy combining **Random/Grid Search** with **NeuroEvolution** to automatically discover optimal CNN architectures for music genre classification from spectrograms.

### Key Features

- **Resource-Efficient**: Designed for 16GB RAM with partial training and memory management
- **Hybrid Strategy**: Random search for exploration + neuroevolution for refinement
- **Comprehensive Logging**: Full traceability of all models and experiments
- **Modular Design**: Easy to customize and extend
- **Production-Ready**: Includes final model evaluation and reporting

---

## Architecture

### Pipeline Stages

1. **Random/Grid Search Phase**
   - Explores hyperparameter space with random sampling
   - Quick evaluation with partial training (10 epochs)
   - Identifies promising architecture regions

2. **NeuroEvolution Phase**
   - Starts from best random search result
   - Iterative refinement through mutation operators
   - Population-based search with survival selection
   - Configurable generations and population size

3. **Full Training Phase**
   - Best model trained with full epochs (50)
   - Early stopping to prevent overfitting
   - Model saved for production use

4. **Final Evaluation**
   - Comprehensive test set evaluation
   - Performance report generation
   - Confusion matrix and metrics

---

## File Structure

```
project_root/
│
├── model_optimization.py          # Main optimization pipeline
├── model_building.py               # build_and_compile_model_03()
├── model_training.py               # train_model_pipeline_04(), evaluate_model_on_test()
├── data_pretreat.py               # Preprocessing functions
│
├── data/
│   └── tfrecords/                 # Preprocessed TFRecords
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── dataset_split.csv
│       └── normalization_stats.json
│
└── optimization_results/          # Output directory (auto-created)
    ├── models/                    # Saved model files
    ├── logs/                      # Detailed logs
    ├── optimization_log.csv       # All evaluated models
    ├── generation_log.csv         # Evolution history
    └── final_report.txt           # Comprehensive final report
```

---

## Configuration

### OptimizationConfig Parameters

```python
@dataclass
class OptimizationConfig:
    # Data paths
    tfrecord_dir: str = "data/tfrecords"
    
    # Resource constraints
    batch_size: int = 32                    # Adjust for your RAM
    partial_training_epochs: int = 10       # Quick evaluation
    full_training_epochs: int = 50          # Final training
    early_stopping_patience: int = 5
    
    # Random search
    random_search_iterations: int = 10      # Number of random configs
    
    # NeuroEvolution
    population_size: int = 8                # Models per generation (4-16)
    num_generations: int = 5                # Evolution cycles
    mutation_rate: float = 0.3              # Mutation probability
    selection_ratio: float = 0.5            # Survival rate (top 50%)
    
    # Output
    output_dir: str = "optimization_results"
    
    # Reproducibility
    random_seed: int = 42
```

### Hyperparameter Search Space

The pipeline explores the following hyperparameters:

- **Convolutional Layers**: 2-4 layers
- **Conv Filters**: [16, 32, 64, 128]
- **Kernel Sizes**: [(3,3), (5,5), (7,7)]
- **Pool Sizes**: [(2,2), (3,3)]
- **Conv Activations**: ['relu', 'elu', 'selu']
- **Dense Layers**: 1-3 layers
- **Dense Neurons**: [64, 128, 256, 512]
- **Dense Activations**: ['relu', 'elu', 'selu']
- **Dropout Rates**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- **Learning Rates**: [0.0001, 0.0005, 0.001, 0.005]

---

## Usage

### Basic Usage

```python
from model_optimization import OptimizationConfig, OptimizationPipeline

# Create configuration
config = OptimizationConfig(
    tfrecord_dir="data/tfrecords",
    batch_size=32,
    partial_training_epochs=10,
    full_training_epochs=50,
    random_search_iterations=10,
    population_size=8,
    num_generations=5,
    output_dir="optimization_results"
)

# Run pipeline
pipeline = OptimizationPipeline(config)
pipeline.run()
```

### Command Line Execution

```bash
python model_optimization.py
```

### Custom Configuration

```python
# For faster testing (fewer iterations)
config_fast = OptimizationConfig(
    random_search_iterations=5,
    population_size=4,
    num_generations=3,
    partial_training_epochs=5
)

# For thorough search (more compute)
config_thorough = OptimizationConfig(
    random_search_iterations=20,
    population_size=16,
    num_generations=10,
    partial_training_epochs=15
)

# For limited RAM (4-8GB)
config_lowram = OptimizationConfig(
    batch_size=16,
    population_size=4,
    cache_dataset=False
)
```

---

## Mutation Operators

The neuroevolution phase uses probabilistic mutations:

### High Probability (Applied frequently)
- **Learning Rate Mutation**: ×0.5 to ×2.0 factor
- **Dropout Mutation**: Change dropout rate

### Medium Probability (Moderate changes)
- **Conv Filter Mutation**: Change number of filters
- **Kernel Size Mutation**: Change kernel dimensions
- **Dense Neuron Mutation**: Change layer size

### Low Probability (Structural changes)
- **Add Conv Layer**: Add new convolutional layer
- **Remove Conv Layer**: Remove existing conv layer
- **Add Dense Layer**: Add new dense layer
- **Remove Dense Layer**: Remove existing dense layer

---

## Output Files

### 1. optimization_log.csv

Complete log of ALL evaluated models with:
- Model ID and generation
- Architecture details (conv_layers, dense_layers, etc.)
- Hyperparameters (learning_rate, dropout, etc.)
- Performance metrics (val_acc, val_loss, train_acc, train_loss)
- Parent model information

**Usage**: Analyze all experiments, compare architectures, identify patterns

### 2. generation_log.csv

Evolution history tracking:
- Generation number
- Best validation accuracy per generation
- Average validation accuracy per generation
- Best model ID

**Usage**: Visualize evolution progress, identify convergence

### 3. final_report.txt

Comprehensive report including:
- Configuration summary
- Best model architecture
- Layer-by-layer details
- Complete performance metrics (train/val/test)
- Total models evaluated

**Usage**: Summary for stakeholders, documentation

### 4. models/ directory

Saved TensorFlow models:
- `best_model_final_YYYYMMDD_HHMMSS.keras`

**Usage**: Load for inference, deployment, further training

---

## Memory Management

The pipeline implements several strategies for 16GB RAM:

1. **Partial Training**: Models trained with few epochs during search
2. **Memory Clearing**: `tf.keras.backend.clear_session()` after each evaluation
3. **Moderate Batch Size**: Default 32 (adjustable)
4. **No Dataset Caching**: Option to disable caching for large datasets
5. **Sequential Evaluation**: Models evaluated one at a time
6. **Reduced Shuffle Buffer**: Smaller buffer for memory efficiency

### Tips for Different RAM Sizes

**4-8GB RAM**:
```python
config = OptimizationConfig(
    batch_size=16,
    population_size=4,
    shuffle_buffer_size=500
)
```

**16-32GB RAM** (default):
```python
config = OptimizationConfig(
    batch_size=32,
    population_size=8,
    shuffle_buffer_size=1000
)
```

**32GB+ RAM**:
```python
config = OptimizationConfig(
    batch_size=64,
    population_size=16,
    cache_dataset=True,
    shuffle_buffer_size=2000
)
```

---

## Workflow Example

### Complete Optimization Run

```
1. Random Search Phase (10 iterations × 10 epochs)
   → Explores diverse architectures
   → Time: ~30-60 minutes
   
2. NeuroEvolution Phase (5 generations × 8 models × 10 epochs)
   → Refines best architecture
   → Time: ~2-4 hours
   
3. Full Training (1 model × 50 epochs)
   → Trains best model completely
   → Time: ~30-60 minutes
   
4. Test Evaluation + Report Generation
   → Final metrics on held-out test set
   → Time: ~5-10 minutes

Total Time: ~3-6 hours (depends on hardware and data size)
```

---

## Interpreting Results

### Understanding the Logs

**optimization_log.csv**:
- Sort by `val_accuracy` to find best models
- Compare architectures: deeper vs wider networks
- Analyze hyperparameter impact: learning rate, dropout

**Visual Analysis** (optional):
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("optimization_results/optimization_log.csv")

# Plot generation progress
df_gen = df.groupby('generation')['val_accuracy'].agg(['max', 'mean'])
df_gen.plot(title="Evolution Progress")
plt.ylabel("Validation Accuracy")
plt.xlabel("Generation")
plt.show()

# Best models by architecture depth
df['num_conv_layers'] = df['conv_layers_str'].apply(lambda x: len(eval(x)))
df.groupby('num_conv_layers')['val_accuracy'].max().plot(kind='bar')
plt.title("Best Performance by Architecture Depth")
plt.show()
```

---

## Customization

### Adding New Mutation Operators

```python
@staticmethod
def mutate_activation_function(config: ModelConfig, mutation_rate: float) -> ModelConfig:
    """Change activation function in a layer."""
    if random.random() < mutation_rate * 0.5:
        idx = random.randint(0, len(config.conv_activations) - 1)
        config.conv_activations[idx] = random.choice(['relu', 'elu', 'selu', 'gelu'])
        log.debug(f"Mutated activation at layer {idx}")
    return config

# Add to MutationOperators.mutate()
mutated = MutationOperators.mutate_activation_function(mutated, mutation_rate)
```

### Custom Hyperparameter Space

```python
class CustomHyperparameterSpace(HyperparameterSpace):
    # Override with your custom ranges
    CONV_FILTERS = [32, 64, 128, 256, 512]  # More options
    KERNEL_SIZES = [(1,1), (3,3), (5,5), (7,7), (9,9)]  # Include 1x1 and larger
    LEARNING_RATES = [0.00005, 0.0001, 0.0005, 0.001]  # Different range
```

### Different Selection Strategy

```python
def select_survivors_tournament(self, population: List[ModelConfig]) -> List[ModelConfig]:
    """Tournament selection instead of top-k."""
    survivors = []
    tournament_size = 3
    
    for _ in range(int(len(population) * self.config.selection_ratio)):
        # Random tournament
        candidates = random.sample(population, tournament_size)
        winner = max(candidates, key=lambda x: x.val_accuracy)
        survivors.append(winner)
    
    return survivors
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```python
# Solution: Reduce batch size and population
config = OptimizationConfig(
    batch_size=16,  # Lower batch size
    population_size=4,  # Fewer models per generation
)
```

**2. Slow Training**
```python
# Solution: Reduce epochs and iterations
config = OptimizationConfig(
    partial_training_epochs=5,  # Faster evaluation
    random_search_iterations=5,  # Fewer iterations
    num_generations=3  # Fewer generations
)
```

**3. Models Not Improving**
```python
# Solution: Increase mutation rate or expand search space
config = OptimizationConfig(
    mutation_rate=0.5,  # More aggressive mutations
    population_size=12,  # Larger population
)
```

**4. Invalid Architectures**
- Check MIN/MAX layer constraints in HyperparameterSpace
- Ensure dropout_rates length matches conv_layers length
- Verify all activations are valid Keras functions

---

## Best Practices

1. **Start Small**: Begin with few iterations/generations to test the pipeline
2. **Monitor Memory**: Use `htop` or Task Manager to watch RAM usage
3. **Check Data**: Ensure TFRecords are correctly preprocessed
4. **Save Frequently**: Pipeline auto-saves, but keep backups
5. **Analyze Logs**: Review optimization_log.csv to understand what works
6. **Iterate**: Use insights from first run to refine search space


## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{model_optimization_pipeline_2025,
  title={Automatic Model Optimization Pipeline for Music Genre Classification},
  author={Colin Manyri},
  year={2025},
  description={Hybrid Random Search and NeuroEvolution for CNN Architecture Search}
}
```

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: colin.manyri@gmail.com

---

## Changelog

### Version 1.0 (2025-12-27)
- Initial release
- Random/Grid Search implementation
- NeuroEvolution with mutation operators
- Comprehensive logging and reporting
- Memory-efficient design for 16GB RAM