import copy
import uuid
import random
from typing import List, Dict

import tensorflow as tf

from src.models.model import build_cnn_model
from src.cste import NUM_CLASSES
from src.logger import get_logger

log = get_logger("model_generator")


class ModelGenerator:
    """
    Generate baseline and mutated CNN models for genetic optimization.
    """

    def __init__(
        self,
        input_shape: tuple,
        mutation_rate: float = 0.3,
        random_seed: int | None = None
    ):
        """
        Args:
            input_shape: Shape of model input (H, W, C)
            mutation_rate: Probability of mutating each hyperparameter
            random_seed: Optional seed for reproducibility
        """
        self.input_shape = input_shape
        self.mutation_rate = mutation_rate

        if random_seed is not None:
            random.seed(random_seed)

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    def initialize_baseline(self, baseline_hyperparams: Dict) -> List[Dict]:
        """
        Create the baseline model (called once).
        """
        model = build_cnn_model(
            input_shape=self.input_shape,
            num_classes=NUM_CLASSES,
            hyperparams=baseline_hyperparams
        )

        model_desc = {
            "model_id": self._generate_id(),
            "generation": 0,
            "hyperparams": baseline_hyperparams,
            "model": model,
            "is_baseline": True
        }

        log.info("Baseline model initialized")

        return [model_desc]

    # --------------------------------------------------
    # Population generation
    # --------------------------------------------------

    def generate_models(
        self,
        parent_models: List[Dict],
        n_mutants: int,
        generation: int
    ) -> List[Dict]:
        """
        Generate mutated models from parent models.
        """
        new_population = []

        for _ in range(n_mutants):
            parent = random.choice(parent_models)
            mutated_hparams = self.mutate_hyperparams(parent["hyperparams"])

            model = build_cnn_model(
                input_shape=self.input_shape,
                num_classes=NUM_CLASSES,
                hyperparams=mutated_hparams
            )

            model_desc = {
                "model_id": self._generate_id(),
                "generation": generation,
                "hyperparams": mutated_hparams,
                "model": model,
                "is_baseline": False
            }

            new_population.append(model_desc)

        log.info(f"{len(new_population)} models generated for generation {generation}")

        return new_population

    # --------------------------------------------------
    # Mutation logic
    # --------------------------------------------------

    def mutate_hyperparams(self, hyperparams: Dict) -> Dict:
        """
        Apply random mutations to hyperparameters.
        """
        mutated = copy.deepcopy(hyperparams)

        for key, value in hyperparams.items():
            if random.random() > self.mutation_rate:
                continue

            mutated[key] = self._mutate_value(key, value)

        return mutated

    def _mutate_value(self, key: str, value):
        """
        Mutation rules per hyperparameter.
        """
        if isinstance(value, int):
            delta = random.choice([-1, 1])
            return max(1, value + delta)

        if isinstance(value, float):
            factor = random.uniform(0.7, 1.3)
            return round(value * factor, 6)

        if isinstance(value, bool):
            return not value

        if isinstance(value, list):
            return random.choice(value)

        return value

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------

    def _generate_id(self) -> str:
        """
        Generate unique model identifier.
        """
        return str(uuid.uuid4())
