"""
Majeure Data Science – UP3 – Machine Learning

TP Bonus – SVM à Marge Douce

Eléves :    Dong-Min KIM
            Gabriel TEIXEIRA LACERDA
            João Pedro MARÇAL STORINO

Enseignant : Youssef SALMAN

Script principal – expériences numériques
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from generate_donnees import generate_clean_dataset, generate_noisy_dataset
from plot_donnees import plot_dataset, plot_decision_boundary


# Cas quasi séparables (jeu clean)

def run_part_clean():
    """
    - Génère le jeu de données clean quasi parfaitement séparables.
    - Entraîne un SVM linéaire avec un C très grand (≈ marge dure).
    - Sauvegarde la figure avec la frontière de décision.
    """
    # jeu clean
    X_clean, y_clean = generate_clean_dataset()

    # SVM linéaire avec C très grand
    clf = SVC(kernel="linear", C=1e6)
    clf.fit(X_clean, y_clean)

    # frontière de décision + marges + vecteurs de support
    plot_decision_boundary(
        X_clean,
        y_clean,
        clf,
        title="SVM linéaire (C = 10^6) – jeu clean (≈ marge dure)",
        filename="fig_clean_boundary.png",
        show=False,
    )

# Cas bruité : influence qualitative de C

def run_part_noisy_qualitative():
    """
    - Génère le jeu de données bruité.
    - Entraîne plusieurs SVM linéaires pour différentes valeurs de C.
    - Produit des figures montrant l'effet de C sur la frontière de décision.
    """
    X_noisy, y_noisy = generate_noisy_dataset()

    C_values = [0.1, 1.0, 100.0]

    for C in C_values:
        clf = SVC(kernel="linear", C=C)
        clf.fit(X_noisy, y_noisy)

        fname = f"fig_noisy_boundary_C{str(C).replace('.', '_')}.png"
        title = f"SVM linéaire sur jeu bruité – C = {C}"

        plot_decision_boundary(
            X_noisy,
            y_noisy,
            clf,
            title=title,
            filename=fname,
            show=False,
        )


# Cas bruité : influence quantitative de C

def run_part_noisy_quantitative():
    """
    - Sépare le jeu bruité en apprentissage / test.
    - Standardise les features.
    - Fait varier C et mesure accuracy en apprentissage et en test,
      ainsi que le nombre de vecteurs de support.
    - Produit une courbe accuracy_test en fonction de C.
    """
    X, y = generate_noisy_dataset()

    # séparation train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    acc_train_list = []
    acc_test_list = []
    n_sv_list = []

    for C in C_values:
        clf = SVC(kernel="linear", C=C)
        clf.fit(X_train_scaled, y_train)

        # Accuracy en apprentissage
        y_pred_train = clf.predict(X_train_scaled)
        acc_train = accuracy_score(y_train, y_pred_train)

        # Accuracy en test
        y_pred_test = clf.predict(X_test_scaled)
        acc_test = accuracy_score(y_test, y_pred_test)

        # Nombre total de vecteurs de support
        n_sv = int(np.sum(clf.n_support_))

        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        n_sv_list.append(n_sv)

        print(f"C = {C:7.2g} | acc_train = {acc_train:.3f} | acc_test = {acc_test:.3f} | #SV = {n_sv}")

    # courbe accuracy_test vs C (échelle log)
    plt.figure(figsize=(6, 5))
    plt.semilogx(C_values, acc_test_list, marker="o")
    plt.xlabel("C (échelle logarithmique)")
    plt.ylabel("Accuracy en test")
    plt.title("Influence du paramètre C sur la performance en test (jeu bruité)")
    plt.grid(True)
    plt.savefig("fig_accuracy_vs_C.png", dpi=300, bbox_inches="tight")
    plt.close()

# point d'entrée principal

if __name__ == "__main__":
    # Tu peux commenter/décommenter selon ce que tu veux générer.
    run_part_clean()
    run_part_noisy_qualitative()
    run_part_noisy_quantitative()
