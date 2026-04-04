import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from db_utils import get_engine

USELESS_COLS = ['EmployeeCount', 'Over18', 'StandardHours']

BINARY_MAP = {
    'Gender':   {'Male': 1, 'Female': 0},
    'OverTime': {'Yes': 1, 'No': 0},
    'Attrition': {'Yes': 1, 'No': 0}
}

MULTI_CAT_COLS = [
    'BusinessTravel', 'Department', 'EducationField',
    'JobRole', 'MaritalStatus'
]

SCALE_COLS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

def load_raw_data() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw_employees", con=engine)
    print(f"Loaded {len(df)} rows from DB.")
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=USELESS_COLS, errors='ignore')
    return df

def encode(df: pd.DataFrame) -> pd.DataFrame:
    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)
    return df

def scale(df: pd.DataFrame, scaler=None, fit=True):
    cols_present = [c for c in SCALE_COLS if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
        return df, scaler
    else:
        df[cols_present] = scaler.transform(df[cols_present])
        return df, scaler

def apply_smote(X: pd.DataFrame, y: pd.Series):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"After SMOTE — class 0: {sum(y_res==0)}, class 1: {sum(y_res==1)}")
    return X_res, y_res

def save_artifacts(scaler, feature_cols, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{output_dir}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"Saved scaler and feature columns to {output_dir}/")

def run_pipeline():
    df = load_raw_data()
    df = clean(df)
    df = encode(df)

    X = df.drop(columns=['Attrition', 'EmployeeNumber'], errors='ignore')
    y = df['Attrition']

    X, scaler = scale(X, fit=True)
    X_res, y_res = apply_smote(X, y)

    save_artifacts(scaler, list(X.columns))

    print(f"Pipeline complete. Final shape: {X_res.shape}")
    return X_res, y_res, scaler, list(X.columns)

if __name__ == "__main__":
    X, y, scaler, features = run_pipeline()