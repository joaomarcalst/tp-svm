import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from sklearn.svm import SVC

from generate_donnees import generate_clean_dataset, generate_noisy_dataset

def plot_dataset(X, y, title: str = "", filename: str | None = None, show: bool = True):
    """
    trace simplement le nuage de points (X[:,0], X[:,1]) coloré par classe.

    cette fonction est utile pour présenter le jeu de données dans le rapport.
    """
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=ListedColormap(["tab:blue", "tab:orange"]),
        edgecolors="k",
        alpha=0.8,
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title if title else "Jeu de données")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["Classe 0", "Classe 1"], loc="best")
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def plot_decision_boundary(X, y, clf, title: str = "", filename: str | None = None, show: bool = True):
    """
    trace la frontière de décision d'un SVM linéaire dans le plan (X[:,0], X[:,1]),
    ainsi que les marges et les vecteurs de support.

    cette fonction sera utilisée pour illustrer :
    - le cas quasi séparables, jeu "clean", C très grand,
    - le cas bruité avec différentes valeurs de C, marge douce.
    """
    # définition d'une grille régulière couvrant le nuage de points
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    # prédictions du modèle sur la grille
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_pred = clf.predict(grid_points)
    Z_pred = Z_pred.reshape(xx.shape)

    # valeur de la fonction de décision sur la grille
    Z_dec = clf.decision_function(grid_points)
    Z_dec = Z_dec.reshape(xx.shape)

    plt.figure(figsize=(6, 5))

    # fond coloré indiquant les régions prédites par le SVM
    plt.contourf(
        xx,
        yy,
        Z_pred,
        alpha=0.2,
        cmap=ListedColormap(["tab:blue", "tab:orange"]),
    )

    # courbes de niveau de la fonction de décision : marge (-1, +1) et frontière (0)
    plt.contour(
        xx,
        yy,
        Z_dec,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        colors="k",
    )

    # nuage de points d'entraînement
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=ListedColormap(["tab:blue", "tab:orange"]),
        edgecolors="k",
        alpha=0.8,
    )

    # vecteurs de support en surbrillance
    if hasattr(clf, "support_vectors_"):
        sv = clf.support_vectors_
        plt.scatter(
            sv[:, 0],
            sv[:, 1],
            s=120,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            label="Vecteurs de support",
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title if title else "SVM linéaire – frontière de décision")

    # légende : classes, vecteurs de support
    class_handles, _ = scatter.legend_elements()
    legend_elements = [
        class_handles[0],  # Classe 0
        class_handles[1],  # Classe 1
        Line2D([], [], color="k", linestyle="-", label="Frontière / marge"),
        Line2D([], [], color="red", marker="o", linestyle="", label="Vecteurs de support"),
    ]
    legend_labels = ["Classe 0", "Classe 1", "Frontière / marge", "Vecteurs de support"]
    plt.legend(legend_elements, legend_labels, loc="best")

    plt.grid(True)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# tracer uniquement les nuages de points sans SVM
if __name__ == "__main__":
    # Jeu clean
    X_clean, y_clean = generate_clean_dataset()
    plot_dataset(
        X_clean,
        y_clean,
        title="Jeu de données clean",
        filename="fig_clean_points.png",
        show=True,
    )

    # jeu bruité
    X_noisy, y_noisy = generate_noisy_dataset()
    plot_dataset(
        X_noisy,
        y_noisy,
        title="Jeu de données bruité",
        filename="fig_noisy_points.png",
        show=True,
    )
