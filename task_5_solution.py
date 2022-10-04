import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# 1
def prepare_data(df: pd.DataFrame):
    return [df.drop(columns=['isFraud', 'TransactionID']), df.isFraud]

# 2
def fit_first_model(X, y, x_test, y_test):
    train_x, validation_x, train_y, validation_y = train_test_split(X.fillna(0), y.fillna(0), train_size=0.7, random_state=1, shuffle=True)
    model = DecisionTreeClassifier(random_state=1)
    model.fit(train_x, train_y)
    validation_score = roc_auc_score(model.predict(validation_x), validation_y)
    test_score = roc_auc_score(model.predict(x_test.fillna(0)), y_test.fillna(0))
    return [round(validation_score, 4), round(test_score, 4)]

# 3
def fit_second_model(X, y, x_test, y_test):
    fill_value = -9999
    train_x, validation_x, train_y, validation_y = train_test_split(X.fillna(fill_value), y, train_size=0.7, random_state=1, shuffle=True)
    model = DecisionTreeClassifier(random_state=1)
    model.fit(train_x, train_y)
    validation_score = roc_auc_score(model.predict(validation_x), validation_y)
    test_score = roc_auc_score(model.predict(x_test.fillna(fill_value)), y_test)
    return [round(validation_score, 4), round(test_score, 4)]

# 4
def model_with_optimal_depth(X, y):
    fill_value = -9999
    cv = KFold(n_splits=5, random_state=27, shuffle=True)

    scores = {}
    for max_depth in range(3, 12):
        model = DecisionTreeClassifier(random_state=1, max_depth=max_depth)
        score = cross_val_score(cv=cv, X=X.fillna(fill_value), y=y, estimator=model).mean()
        scores[score] = max_depth

    return scores[max(scores.keys())]

# 5
def model_with_optimal_samples(X, y):
    fill_value = -9999
    cv = KFold(n_splits=5, random_state=27, shuffle=True)

    scores = {}
    for leaf_count in [5, 10, 15, 25, 50, 100, 150, 250, 500, 1000, 2500, 5000, 10000]:
        model = DecisionTreeClassifier(random_state=1, max_depth=10, max_leaf_nodes=leaf_count)
        score = cross_val_score(cv=cv, X=X.fillna(fill_value), y=y, estimator=model).mean()
        scores[score] = leaf_count

    return scores[max(scores.keys())]

# 6
def model_with_optimal_features(X, y):
    fill_value = -9999
    cv = KFold(n_splits=5, random_state=27, shuffle=True)

    scores = {}
    for features in range(1, X.shape[1]):
        model = DecisionTreeClassifier(random_state=1, max_depth=10, max_leaf_nodes=100, max_features=features)
        score = cross_val_score(cv=cv, X=X.fillna(fill_value), y=y, estimator=model).mean()
        scores[score] = features

    return scores[max(scores.keys())]

# 7
def model_with_optimal_params(X, y):
    cv = KFold(n_splits=5, random_state=27, shuffle=True)
    gs = GridSearchCV(DecisionTreeClassifier(random_state = 1),
                      cv = cv,
                      param_grid = {
                          'max_features': range(1, X.shape[1]),
                          'max_depth': range(3, 12),
                          'min_samples_leaf': [5, 10, 15, 25, 50, 100, 150, 250, 500, 1000, 2500, 5000, 10000],
                          'criterion': ['entropy', 'gini']
                      },
                      scoring='r2')
    gs.fit(X.fillna(-9999), y)
    return gs.cv_results_['params']

# 8
def fit_final_model(X, y, x_test, y_test):
    fill_value = -9999
    train_x, validation_x, train_y, validation_y = train_test_split(X.fillna(fill_value), y, train_size=0.7, random_state=1, shuffle=True)
    model = DecisionTreeClassifier(random_state=1, max_features=13, max_depth=11, min_samples_leaf=5)
    model.fit(train_x, train_y)
    validation_score = roc_auc_score(model.predict(validation_x), validation_y)
    test_score = roc_auc_score(model.predict(x_test.fillna(fill_value)), y_test)
    return [round(validation_score, 4), round(test_score, 4)]

#9
def check_tree_stability(X, y, x_test, y_test):
    fill_value = -9999
    val_scores = np.array([])
    test_scores = np.array([])
    for i in range(5):
        indecies = np.random.randint(X.shape[0], size=int(X.shape[0]))
        leng = round(len(indecies) * 0.9)
        match = indecies[:leng]
        not_match = indecies[leng:]

        tr_x = X.fillna(fill_value).iloc[match]
        tr_y = y.iloc[match]

        val_x = X.fillna(fill_value).iloc[not_match]
        val_y = y.iloc[not_match]

        model = DecisionTreeClassifier(random_state=1, max_features=13, max_depth=11, min_samples_leaf=5)
        model.fit(tr_x, tr_y)
        val_scores = np.append(val_scores, roc_auc_score(model.predict(val_x), val_y))
        test_scores = np.append(test_scores, roc_auc_score(model.predict(x_test.fillna(fill_value)), y_test))

    return [val_scores.mean(), val_scores.std(), test_scores.mean(), test_scores.std()]