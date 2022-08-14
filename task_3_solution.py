import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

def split_data_into_two_samples(df: pd.DataFrame,
                                ranom_state: int = 42):
    return train_test_split(df, train_size=0.7, random_state=ranom_state, shuffle=True)

def prepare_data(df: pd.DataFrame):
    return (df.select_dtypes([np.number]).dropna(axis=1).drop(columns=["price_doc", "id"]),
            df["price_doc"])

def scale_data(df: pd.DataFrame, transformer: TransformerMixin):
    scaled = transformer.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)

def prepare_data_for_model(df: pd.DataFrame, transformer: TransformerMixin):
    X_train, y_train = prepare_data(df)
    X_train_scaled = scale_data(X_train, transformer)
    return X_train_scaled, y_train

def fit_first_linear_model(x_train, y_train):
    return LinearRegression().fit(x_train, y_train)

# X_train, X_test = split_data_into_two_samples(data)

# X_train_scaled_1, y_train_1 = prepare_data_for_model(X_train, StandardScaler())
# X_test_scaled_1, y_test_1 = prepare_data_for_model(X_test, StandardScaler())
# model1 = fit_first_linear_model(X_train_scaled_1, y_train_1)

# X_train_scaled_2, y_train_2 = prepare_data_for_model(X_train, MinMaxScaler())
# X_test_scaled_2, y_test_2 = prepare_data_for_model(X_test, MinMaxScaler())
# model2 = fit_first_linear_model(X_train_scaled_2, y_train_2)

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    MSE = round(mean_squared_error(y_test, y_pred), 2)
    MAE = 123 # round(mean_absolute_error(y_test, y_pred), 2)
    R2 = round(r2_score(y_test, y_pred), 2)
    return [MSE, MAE, R2]

def calculate_model_weights(model, features):
    sorted_weights = sorted(zip(model.coef_, features), reverse=True)
    weights = pd.Series([x[0] for x in sorted_weights])
    features = pd.Series([x[1] for x in sorted_weights])
    df = pd.DataFrame({'features': features, 'weights': weights})
    return df