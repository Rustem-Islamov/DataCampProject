import os

import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import KFold
import ipdb
import math

problem_title = "French Presidential Elections"
_target_column_name = "Voix"
_name_candidate = "MACRON"
columns_to_be_used = [
    "Code du département",
    "Libellé du département",
    "Code de la circonscription",
    "Libellé de la circonscription",
    "Code de la commune",
    "Libellé de la commune",
    "Code du b.vote",
    "Inscrits",
    "Votants",
    "location",
]
_excluded_territories = [
    "Français établis hors de France",
    "Wallis et Futuna",
    "Guyane",
    "Saint-Pierre-et-Miquelon",
    "Mayotte",
    "Saint-Martin/Saint-Barthélemy",
    "Polynésie française",
    "Nouvelle-Calédonie",
]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.EstimatorExternalData()

score_types = [
    rw.score_types.RMSE(name="rmse", precision=3),
]


def get_cv(X, y, random_state=0):
    cv = KFold(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="."):
    f_name = "elections-france-presidentielles-2022-1er-tour-par-bureau-de-vote.csv"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test_data.csv"
    return _read_data(path, f_name)