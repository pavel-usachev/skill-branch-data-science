import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import BaggingClassifier

# 1


def prepare_numerical_data(data: pd.DataFrame):
    return data.drop(columns=['isFraud', 'TransactionID', 'TransactionDT']), data.isFraud

# 2


def fit_base_model(X, y, x_test, y_test):
    X = X.fillna(0)
    x_test = x_test.fillna(0)

    train_x, valid_x = train_test_split(X,
                                        train_size=0.7,
                                        random_state=1,
                                        shuffle=True)

    train_y, valid_y = train_test_split(y,
                                        train_size=0.7,
                                        random_state=1,
                                        shuffle=True)

    model = DecisionTreeClassifier(criterion='gini',
                                   max_depth=11,
                                   max_features=14,
                                   min_samples_leaf=50,
                                   random_state=1)

    model.fit(train_x, train_y)

    score_valid = roc_auc_score(model.predict(valid_x), valid_y)
    score_test = roc_auc_score(model.predict(x_test), y_test)
    return [round(score_valid, 4), round(score_test, 4)]

# 3


def fit_first_bagging(X, y, x_test, y_test):
    X = X.fillna(0)
    x_test = x_test.fillna(0)

    train_x, valid_x = train_test_split(X,
                                        train_size=0.7,
                                        random_state=1,
                                        shuffle=True)

    train_y, valid_y = train_test_split(y,
                                        train_size=0.7,
                                        random_state=1,
                                        shuffle=True)

    model = DecisionTreeClassifier(criterion='gini',
                                   max_depth=11,
                                   max_features=14,
                                   min_samples_leaf=50,
                                   random_state=1)

    bagging = BaggingClassifier(base_estimator=model, random_state=1)

    bagging.fit(train_x.reset_index(drop=True), train_y.reset_index(drop=True))

    score_valid = roc_auc_score(bagging.predict(valid_x), valid_y)
    score_test = roc_auc_score(bagging.predict(x_test), y_test)
    return [round(score_valid, 4), round(score_test, 4)]

# 4
