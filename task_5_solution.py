import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

def prepare_data(df: pd.DataFrame):
    return [df.drop(columns=['isFraud', 'TransactionID']), df.isFraud]

def fit_first_model(X, y, x_test, y_test):
    train_x, validation_x, train_y, validation_y = train_test_split(X.fillna(0), y.fillna(0), train_size=0.7, random_state=1, shuffle=True)
    model = DecisionTreeClassifier(random_state=1)
    model.fit(train_x, train_y)
    validation_score = roc_auc_score(model.predict(validation_x), validation_y)
    test_score = roc_auc_score(model.predict(x_test.fillna(0)), y_test.fillna(0))
    return [round(validation_score, 4), round(test_score, 4)]

def model_with_optimal_depth(X, y):
    fill_value = -9999
    cv = KFold(n_splits=5, random_state=27, shuffle=True)

    scores = {}
    for max_depth in range(3, 12):
        model = DecisionTreeClassifier(random_state=1, max_depth=max_depth)
        score = cross_val_score(cv=cv, X=X.fillna(fill_value), y=y, estimator=model).mean()
        scores[score] = max_depth

    return scores[max(scores.keys())]

def model_with_optimal_samples(X, y):
    fill_value = -9999
    cv = KFold(n_splits=5, random_state=27, shuffle=True)

    scores = {}
    for leaf_count in [5, 10, 15, 25, 50, 100, 150, 250, 500, 1000, 2500, 5000, 10000]:
        model = DecisionTreeClassifier(random_state=1, max_depth=10, max_leaf_nodes=leaf_count)
        score = cross_val_score(cv=cv, X=X.fillna(fill_value), y=y, estimator=model).mean()
        scores[score] = leaf_count

    return scores[max(scores.keys())]

def model_with_optimal_features(X, y):
    fill_value = -9999
    cv = KFold(n_splits=5, random_state=27, shuffle=True)

    scores = {}
    for features in range(1, X.shape[1]):
        model = DecisionTreeClassifier(random_state=1, max_depth=10, max_leaf_nodes=100, max_features=features)
        score = cross_val_score(cv=cv, X=X.fillna(fill_value), y=y, estimator=model).mean()
        scores[score] = features

    return scores[max(scores.keys())]