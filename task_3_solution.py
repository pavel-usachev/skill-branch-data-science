import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def split_data_into_two_samples(df: pd.DataFrame,
                                ranom_state: int = 42):
    return train_test_split(df, train_size=0.7, random_state=ranom_state, shuffle=True)

def prepare_data(df: pd.DataFrame):
    df = df.drop(columns=["id"])
    y = df["price_doc"]
    X = df.select_dtypes([np.number]).dropna(axis=1)
    return X, y

def scale_data(df: pd.DataFrame, transformer):
    return transformer.fit_transform(df);

def prepare_data_for_model(df: pd.DataFrame, transformer):
    X_train, y_train = prepare_data(df)
    X_train_scaled = scale_data(X_train, transformer)
    return X_train_scaled, y_train

X_train, X_test = split_data_into_two_samples(data)
X_train_scaled, y_train = prepare_data_for_model(X_train, StandardScaler())
X_test_scaled, y_test = prepare_data_for_model(X_test, StandardScaler())

def fit_first_linear_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

model1 = fit_first_linear_model(X_train_scaled, y_train)

X_train, X_test = split_data_into_two_samples(data)
X_train_scaled, y_train = prepare_data_for_model(X_train, MinMaxScaler())

model2 = fit_first_linear_model(X_train_scaled, y_train)

def evaluate_model(model: LinearRegression, X_test, y_test):
    y_pred = model.predict(X_test)
    return [round(value, 2) for value in [mean_squared_error(y_test, y_pred),
            mean_absolute_error(y_test, y_pred),
            r2_score(y_test, y_pred)]]

evaluate_model(model1, X_test_scaled, y_test)

def calculate_model_weights(model: LinearRegression, feature_names):
    indicies = [X_train.columns.get_loc(feature) for feature in feature_names]
    weights = model.coef_[indicies]
    sorted_weights = sorted(zip(weights, feature_names), reverse=True)
    return sorted_weights

calculate_model_weights(model1, ["full_sq", "floor"])