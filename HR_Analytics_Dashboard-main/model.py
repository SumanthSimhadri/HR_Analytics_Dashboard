import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    df = pd.read_csv("data.csv")
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    X = df[['Age', 'JobSatisfaction', 'MonthlyIncome', 'OverTime']]
    y = df['Attrition']

    model = RandomForestClassifier()
    model.fit(X, y)
    return model
