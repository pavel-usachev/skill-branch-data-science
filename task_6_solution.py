import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# 1
def prepare_numerical_data(data: pd.DataFrame):
    return data.drop(columns=['isFraud', 'TransactionID', 'TransactionDT']), data.isFraud