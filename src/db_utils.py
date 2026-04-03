import pandas as pd
from sqlalchemy import create_engine

DB_USER = "root"          
DB_PASS = "root"  
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "attrition_db"

def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)

def load_raw_data(csv_path: str):
    df = pd.read_csv(csv_path)
    engine = get_engine()
    df.to_sql(
        name="raw_employees",
        con=engine,
        if_exists="replace",
        index=False
    )
    print(f"Loaded {len(df)} rows into raw_employees table.")
    return df

if __name__ == "__main__":
    df = load_raw_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print(df.head())