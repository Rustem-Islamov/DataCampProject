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


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, "data", f_name), sep=";", low_memory=False)
    data = data[data['Nom']==_name_candidate.upper()]
    data = data[columns_to_be_used + [_target_column_name]]
    data = _filter_data(data)
    data["Code de la commune"] = data["Code de la commune"].astype(str)
    data = data.sort_values(["Code de la commune"])
    y_array = data[_target_column_name].values
    X_df = data.drop(columns=[_target_column_name])
    return X_df, y_array


def _filter_data(data):
    data = data.copy()
    filtered_data = data[(~data["Libellé du département"].isin(_excluded_territories))]
    return filtered_data


def get_latitude_longitude(df):
    df = df.copy()
    df["longitude"] = df["location"].apply(get_longitude)
    df["latitude"] = df["location"].apply(get_latitude)
    df = df.drop(columns=["location"])
    return df


def get_latitude(element):
    try:
        if math.isnan(element):
            return None
    except:
        latitude = element.split(",")[0]
    return float(latitude)


def get_longitude(element):
    try:
        if math.isnan(element):
            return None
    except:
        longitude = element.split(",")[1].strip()
    return float(longitude)


def correct_missing_locations(df):
    df = df.copy()
    df_columns = df.columns
    n_missing = df[df["latitude"].isnull()].shape[0]
    df_group = (
        df.drop_duplicates(["Libellé de la commune"])
        .groupby("Code du département")[["longitude", "latitude"]]
        .mean()
    )
    df_std = df.groupby("Code du département")[["longitude", "latitude"]].std()
    dict_longitude = df_group.iloc[:, 0].to_dict()
    dict_latitude = df_group.iloc[:, 1].to_dict()
    dict_std_longitude = df_std.iloc[:, 0].to_dict()
    dict_std_latitude = df_group.iloc[:, 1].to_dict()
    df["std_longitude"] = df["Code du département"].map(dict_std_longitude)
    df["mean_longitude"] = df["Code du département"].map(dict_longitude)
    df["std_latitude"] = df["Code du département"].map(dict_std_latitude)
    df["mean_latitude"] = df["Code du département"].map(dict_latitude)
    df.loc[df["latitude"].isnull(), "latitude"] = (
        np.random.randn(n_missing) * df.loc[df["latitude"].isnull(), "std_latitude"]
        + df.loc[df["latitude"].isnull(), "mean_latitude"]
    )
    df.loc[df["longitude"].isnull(), "longitude"] = (
        np.random.randn(n_missing) * df.loc[df["longitude"].isnull(), "std_longitude"]
        + df.loc[df["longitude"].isnull(), "mean_longitude"]
    )

    return df[df_columns]
