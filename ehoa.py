"""Enhanced Hiking Optimization Algorithm for binary feature selection.

The implementation follows Eqs. 6, 7, 9, 10, 11 and 12 of Hegazy et al.
Continuous leader positions and stochastic binary masks are deliberately kept
separate; confusing them was a major defect in the original student version.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

from utils import CHAOTIC_MAPS, chaotic_sequence, evaluate_mask_cv, prepare_cv_folds, sigmoid


@dataclass(frozen=True)
class EHOAResult:
    mask: np.ndarray
    selected_features: np.ndarray
    accuracy: float
    fitness: float
    runtime_seconds: float
    evaluations: int


class EHOA:
    """Binary EHOA wrapper feature selector.

    Defaults match the paper's search budget. Set smaller values from the CLI
    with ``--profile quick`` while developing.
    """

    def __init__(
        self,
        n_hikers: int = 30,
        max_iter: int = 50,
        sf_min: float = 1.0,
        sf_max: float = 3.0,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        chaotic_map: str = "tent",
        alpha: float = 0.99,
        n_folds: int = 10,
        position_min: float = -6.0,
        position_max: float = 6.0,
        apply_smote: bool = True,
        random_state: int = 42,
        patience: int | None = None,
        verbose: bool = True,
    ) -> None:
        if n_hikers < 2 or max_iter < 1:
            raise ValueError("n_hikers must be >= 2 and max_iter must be >= 1")
        if chaotic_map not in CHAOTIC_MAPS:
            raise ValueError(f"Unknown chaotic map {chaotic_map!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if position_min >= position_max:
            raise ValueError("position_min must be lower than position_max")

        self.n_hikers = n_hikers
        self.max_iter = max_iter
        self.sf_min = sf_min
        self.sf_max = sf_max
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.chaotic_map = chaotic_map
        self.alpha = alpha
        self.n_folds = n_folds
        self.position_min = position_min
        self.position_max = position_max
        self.apply_smote = apply_smote
        self.random_state = random_state
        self.patience = patience
        self.verbose = verbose
        self._reset_state()

    def _reset_state(self) -> None:
        self.best_solution: np.ndarray | None = None
        self.best_position: np.ndarray | None = None
        self.best_fitness = float("inf")
        self.best_accuracy = 0.0
        self.best_features = np.array([], dtype=int)
        self.convergence_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.feature_count_history: list[int] = []
        self.diversity_history: list[float] = []
        self.evaluations_ = 0
        self.runtime_seconds_ = 0.0
        self.result_: EHOAResult | None = None
        self._fitness_cache: dict[bytes, tuple[float, float, int]] = {}

    def _initialize_population(self, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        values = chaotic_sequence(
            self.chaotic_map,
            self.n_hikers * n_features,
            x0=0.7,
        ).reshape(self.n_hikers, n_features)
        span = self.position_max - self.position_min
        population = self.position_min + values * span
        velocities = self._rng.uniform(-0.1 * span, 0.1 * span, population.shape)
        return population, velocities

    def _adaptive_sweep_factor(self, iteration: int) -> float:
        return self.sf_max - (iteration / self.max_iter) * (self.sf_max - self.sf_min)

    def _inertia_weight(self, iteration: int) -> float:
        return self.w_max - (iteration / self.max_iter) * (self.w_max - self.w_min)

    def _to_mask(self, position: np.ndarray) -> np.ndarray:
        probabilities = sigmoid(position)
        mask = self._rng.random(position.shape) < probabilities
        if not np.any(mask):
            mask[int(np.argmax(probabilities))] = True
        return mask

    def _evaluate(self, mask: np.ndarray) -> tuple[float, float, int]:
        key = np.packbits(mask.astype(np.uint8)).tobytes() + len(mask).to_bytes(4, "little")
        if key not in self._fitness_cache:
            self._fitness_cache[key] = evaluate_mask_cv(mask, self._folds, self.alpha)
            self.evaluations_ += 1
        return self._fitness_cache[key]

    def _consider_global(
        self,
        position: np.ndarray,
        mask: np.ndarray,
        fitness: float,
        accuracy: float,
    ) -> bool:
        if fitness < self.best_fitness:
            self.best_fitness = float(fitness)
            self.best_accuracy = float(accuracy)
            self.best_position = position.copy()
            self.best_solution = mask.astype(np.uint8).copy()
            self.best_features = np.flatnonzero(mask)
            return True
        return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Select features using training data only.

        Returns the legacy tuple ``(mask, cv_accuracy, feature_indices)``.
        Detailed metadata is available as ``result_``.
        """
        started = time.perf_counter()
        self._reset_state()
        self._rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and aligned with y")

        self._folds = prepare_cv_folds(
            X,
            y,
            n_folds=self.n_folds,
            random_state=self.random_state,
            apply_smote=self.apply_smote,
        )
        _, n_features = X.shape
        population, velocities = self._initialize_population(n_features)
        personal_positions = population.copy()
        personal_fitness = np.full(self.n_hikers, np.inf)

        for index in range(self.n_hikers):
            mask = self._to_mask(population[index])
            fitness, accuracy, _ = self._evaluate(mask)
            personal_fitness[index] = fitness
            self._consider_global(population[index], mask, fitness, accuracy)

        stale_iterations = 0
        velocity_limit = self.position_max - self.position_min
        for iteration in range(1, self.max_iter + 1):
            improved = False
            sf = self._adaptive_sweep_factor(iteration)
            inertia = self._inertia_weight(iteration)
            assert self.best_position is not None

            for index in range(self.n_hikers):
                r1 = self._rng.random(n_features)
                r2 = self._rng.random(n_features)
                cognitive = self.c1 * r1 * (personal_positions[index] - population[index])
                social = self.c2 * r2 * (self.best_position - population[index])
                velocities[index] = np.clip(
                    inertia * velocities[index] + sf * (cognitive + social),
                    -velocity_limit,
                    velocity_limit,
                )
                population[index] = np.clip(
                    population[index] + velocities[index],
                    self.position_min,
                    self.position_max,
                )

                mask = self._to_mask(population[index])
                fitness, accuracy, _ = self._evaluate(mask)
                if fitness < personal_fitness[index]:
                    personal_fitness[index] = fitness
                    personal_positions[index] = population[index].copy()
                improved |= self._consider_global(
                    population[index], mask, fitness, accuracy
                )

            self.convergence_history.append(self.best_fitness)
            self.accuracy_history.append(self.best_accuracy)
            self.feature_count_history.append(len(self.best_features))
            self.diversity_history.append(float(np.mean(np.std(population, axis=0))))
            stale_iterations = 0 if improved else stale_iterations + 1

            if self.verbose and (iteration == 1 or iteration % 10 == 0 or iteration == self.max_iter):
                print(
                    f"iteration={iteration:03d}/{self.max_iter} "
                    f"fitness={self.best_fitness:.5f} "
                    f"cv_accuracy={self.best_accuracy:.4f} "
                    f"features={len(self.best_features)}/{n_features}"
                )
            if self.patience is not None and stale_iterations >= self.patience:
                if self.verbose:
                    print(f"early_stop: no improvement for {self.patience} iterations")
                break

        self.runtime_seconds_ = time.perf_counter() - started
        assert self.best_solution is not None
        self.result_ = EHOAResult(
            mask=self.best_solution.copy(),
            selected_features=self.best_features.copy(),
            accuracy=self.best_accuracy,
            fitness=self.best_fitness,
            runtime_seconds=self.runtime_seconds_,
            evaluations=self.evaluations_,
        )
        return self.best_solution.copy(), self.best_accuracy, self.best_features.copy()

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "iteration": np.arange(1, len(self.convergence_history) + 1),
                "fitness": self.convergence_history,
                "cv_accuracy": self.accuracy_history,
                "selected_features": self.feature_count_history,
                "population_diversity": self.diversity_history,
            }
        )


def test_chaotic_maps(
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int = 3,
    *,
    n_hikers: int = 12,
    max_iter: int = 15,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Compare all maps with independent, reproducible runs."""
    results: dict[str, dict[str, float]] = {}
    for map_name in CHAOTIC_MAPS:
        accuracies, fitnesses, feature_counts = [], [], []
        for run in range(n_runs):
            selector = EHOA(
                chaotic_map=map_name,
                n_hikers=n_hikers,
                max_iter=max_iter,
                n_folds=n_folds,
                random_state=random_state + run,
                verbose=False,
            )
            _, accuracy, features = selector.fit(X, y)
            accuracies.append(accuracy)
            fitnesses.append(selector.best_fitness)
            feature_counts.append(len(features))
        results[map_name] = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies, ddof=1)) if n_runs > 1 else 0.0,
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses, ddof=1)) if n_runs > 1 else 0.0,
            "mean_features": float(np.mean(feature_counts)),
            "std_features": float(np.std(feature_counts, ddof=1)) if n_runs > 1 else 0.0,
        }
    return results
