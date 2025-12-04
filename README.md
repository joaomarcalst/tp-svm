# TP Bonus – SVM à marge douce

Ce dépôt contient le code utilisé pour le TP bonus de la Majeure Data Science (UP3 – Machine Learning) sur le thème **SVM à marge douce**.

## Contenu des fichiers

- `generate_donnees.py`  
  Génération des jeux de données synthétiques :
  - jeu *clean* (données quasi linéairement séparables, sans bruit) ;
  - jeu *bruité* (avec bruit de labellisation), utilisé pour illustrer la marge douce.

- `plot_donnees.py`  
  Fonctions de visualisation :
  - `plot_dataset` : trace le nuage de points (classes 0 et 1) ;
  - `plot_decision_boundary` : trace la frontière de décision d’un SVM linéaire, les marges et les vecteurs de support.

- `tp_bonus.py`  
  Script principal qui orchestre les expériences décrites dans le rapport :
  - cas quasi séparables (jeu *clean*, comportement proche de la marge dure) ;
  - cas bruité, influence qualitative de C (frontières pour différentes valeurs de C) ;
  - cas bruité, influence quantitative de C (accuracy en fonction de C).

## Prérequis

- Python 3.x  
- Bibliothèques principales :
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

L’installation peut se faire par exemple avec :

```bash
pip install numpy matplotlib scikit-learn
```

## Exécution

Depuis le répertoire du TP :

- Pour générer les nuages de points des jeux de données :

```bash
python plot_donnees.py
```

- Pour générer les principales figures utilisées dans le rapport (cas *clean* et cas bruité) :

```bash
python tp_bonus.py
```

Les figures sont sauvegardées sous forme de fichiers `.png` dans le même répertoire.
