import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_data_into_two_samples(x):
    x_train, x_test = train_test_split(x, test_size=0.3, random_state=42)
    return [x_train, x_test]

def prepare_data(x):
    target_vector = x['price_doc']
    objects = x.select_dtypes(include=['object']).columns
    filtered = x.drop(columns=objects).drop(columns=["id"]).drop(columns=['price_doc']).dropna(axis=1)
    return [filtered, target_vector]

def scale_data(x, transformer):
    return transformer.fit_transform(x)

def prepare_data_for_model(x, transformer):
    x_train, x_test = prepare_data(x)
    x_scaled_array = scale_data(x_train, transformer)
    x_train_scaled = pd.DataFrame(x_scaled_array, columns=x_train.columns)
    return [x_train_scaled, x_test]

def fit_first_linear_model(x_train, y_train):
    x_train_scaled = scale_data(x_train, StandardScaler())
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    return model

def evaluate_model(linreg, x_test, y_test):
    y_pred = linreg.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return [round(mse, 2), round(mae, 2), round(r2, 2)]

def calculate_model_weights(linreg, names):
    df = pd.DataFrame({
        'features': names,
        'weights': linreg.coef_
    })
    df.sort_values(by=['weights'])
    return df