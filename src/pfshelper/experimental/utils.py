import pandas as pd

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    
    #reduce size of the dataframe, borrowed from: https://www.kaggle.com/yamqwe/feature-engineering-lgb-blend
    
    float_cols = [c for c in df if df[c].dtype in ["float64"]]
    int_cols = [c for c in df if df[c].dtype in ['int64']]
    df[float_cols] = df[float_cols].astype('float32')
    df[int_cols] = df[int_cols].astype('int16')
    
    return df
    