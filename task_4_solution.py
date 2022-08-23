import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def get_roc_auc(model, x_valid, y_valid, x_test, y_test):
    return [round(score, 4) for score in [roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]), 
                                          roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])]]

def calculate_data_stats(df: pd.DataFrame):
    return [df.shape,
            df.dtypes[(df.dtypes == 'float64') | (df.dtypes == 'int64')].count(),
            df.dtypes[df.dtypes == 'object'].count(),
            round(df["isFraud"].mean() * 100, 2)]

def prepare_data(df: pd.DataFrame):
    return [df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]),
            df["isFraud"]]

def fit_first_model(X, y, x_test, y_test):
    X = X.fillna(0)
    x_test = x_test.fillna(0)

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, random_state=1, shuffle=True)

    model = LogisticRegression(random_state=1).fit(x_train, y_train)
    return get_roc_auc(model, x_valid, y_valid, x_test, y_test)

def fit_second_model(X, y, x_test, y_test):
    X = X.fillna(X.mean())
    x_test = x_test.fillna(x_test.mean())

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, random_state=1, shuffle=True)

    model = LogisticRegression(random_state=1).fit(x_train, y_train)
    return get_roc_auc(model, x_valid, y_valid, x_test, y_test)

def fit_third_model(X, y, x_test, y_test):
    X = X.fillna(X.median())
    x_test = x_test.fillna(x_test.median())

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, random_state=1, shuffle=True)

    model = LogisticRegression(random_state=1).fit(x_train, y_train)
    return get_roc_auc(model, x_valid, y_valid, x_test, y_test)

def fit_fourth_model(X, y, x_test, y_test):
    X = X.fillna(0)
    x_test = x_test.fillna(0)

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, random_state=1, shuffle=True)

    pipeline = Pipeline(steps=[("scaling", StandardScaler()),
                               ("model", LogisticRegression(random_state=1))])
    model = pipeline.fit(x_train, y_train)
    return get_roc_auc(model, x_valid, y_valid, x_test, y_test)

def fit_fifth_model(X, y, x_test, y_test):
    X = X.fillna(X.mean())
    x_test = x_test.fillna(x_test.mean())

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, random_state=1, shuffle=True)

    pipeline = Pipeline(steps=[("scaling", StandardScaler()),
                               ("model", LogisticRegression(random_state=1))])
    model = pipeline.fit(x_train, y_train)
    return get_roc_auc(model, x_valid, y_valid, x_test, y_test)

def fit_sixth_model(X, y, x_test, y_test):
    X = X.fillna(0)
    x_test = x_test.fillna(0)

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, random_state=1, shuffle=True)

    pipeline = Pipeline(steps=[("scaling", MinMaxScaler()),
                               ("model", LogisticRegression(random_state=1))])
    model = pipeline.fit(x_train, y_train)
    return get_roc_auc(model, x_valid, y_valid, x_test, y_test)

def fit_seventh_model(X, y, x_test, y_test):
    X = X.fillna(X.mean())
    x_test = x_test.fillna(x_test.mean())

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.7, random_state=1, shuffle=True)

    pipeline = Pipeline(steps=[("scaling", MinMaxScaler()),
                               ("model", LogisticRegression(random_state=1))])
    model = pipeline.fit(x_train, y_train)
    return get_roc_auc(model, x_valid, y_valid, x_test, y_test)

def find_best_split(X, y, x_test, y_test):
    X = X.fillna(X.median())
    x_test = x_test.fillna(x_test.median())
    result = { "train_size": [], "valid_score": [], "test_score": [] }
    for train_size in [x / 10 for x in range(1, 10)]:
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=train_size, random_state=1, shuffle=True)
        pipeline = Pipeline(steps=[("scaling", MinMaxScaler()),
                                   ("model", LogisticRegression(random_state=1))])
        model = pipeline.fit(x_train, y_train)
        score_valid, score_test = get_roc_auc(model, x_valid, y_valid, x_test, y_test)
        result["train_size"].append(x_train.shape[0])
        result["valid_score"].append(score_valid)
        result["test_score"].append(score_test)
    return pd.DataFrame(result)

def choose_best_split(bs: pd.DataFrame):
    bs["delta"] = bs["test_score"] - bs["valid_score"]
    return bs.sort_values("delta", ignore_index=True)["train_size"][0]