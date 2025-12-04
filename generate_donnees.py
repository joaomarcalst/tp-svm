"""
Majeure Data Science – UP3 – Machine Learning

TP Bonus – SVM à Marge Douce

Ce fichier regroupe les fonctions de génération des jeux de données utilisés
dans le rapport :

- generate_clean_dataset : jeu  "clean" quasi séparables, sans bruit de labellisation,
  utilisé pour illustrer le comportement proche de la marge dure.

- generate_noisy_dataset : jeu bruité, plus réaliste, utilisé pour illustrer le SVM
  à marge douce et l’influence du paramètre C.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification


def generate_clean_dataset(
    n_samples: int = 200,
    class_sep: float = 1.8,
    random_state: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère un jeu de données clean quasi linéairement séparables,
    sans bruit de labellisation (flip_y = 0.0).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=0.0,  # pas de bruit de labellisation
        random_state=random_state,
    )
    return X, y


def generate_noisy_dataset(
    n_samples: int = 300,
    class_sep: float = 1.2,
    flip_y: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    génère un jeu de données bruité, plus réaliste, pour illustrer le SVM à marge douce.

    on introduit un bruit de labellisation via le paramètre flip_y > 0.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )
    return X, y


if __name__ == "__main__":
    # petit test rapide : générer les jeux et afficher leurs dimensions
    X_clean, y_clean = generate_clean_dataset()
    X_noisy, y_noisy = generate_noisy_dataset()

    print(f"Jeu clean : X = {X_clean.shape}, y = {y_clean.shape}, classes = {np.unique(y_clean)}")
    print(f"Jeu bruité : X = {X_noisy.shape}, y = {y_noisy.shape}, classes = {np.unique(y_noisy)}")