import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRegressor
import pickle


df = pd.read_csv("players-stats.csv")
df.columns = df.columns.str.lower().str.replace(" ", "_")


def df_cleaning(df):
    df = df.drop("other_positions", axis=1)
    df = df.dropna(subset=['contract_expiration'])
    df = df.drop("born", axis=1)
    df = df.drop("contract_expiration", axis=1)
    df["value_euros"] = np.log1p(df["value_euros"])
    numerical_final = ["age", "years_remaining", "prgc", "npxg+xag"] 
    categorical_final = ["team", "pos"]
    target = ["value_euros"]
    df_final = df[numerical_final + categorical_final + target]
    return df_final


def validation_framework(df):
    df_full_train, test = train_test_split(df, test_size=0.2, random_state=1)
    y_full = df_full_train.value_euros.values
    df_full_train = df_full_train.drop("value_euros", axis=1)

    return df_full_train, y_full


def prepare_x(df):
    dv = DictVectorizer(sparse=False)

    dictionary = df.to_dict(orient="records")
    x = dv.fit_transform(dictionary)
    return x, dv


def model_training(x, y):
    final_model = XGBRegressor(n_estimators=3000,
                           learning_rate=0.01,
                           max_depth=3,
                           subsample=0.8,
                           colsample_bytree=0.7,
                           objective="reg:squarederror",
                           random_state=42,
                           n_jobs=-1)

    final_model.fit(x, y)
    return final_model




df_final = df_cleaning(df)

df_full_train, y_full_train = validation_framework(df_final)
x_full_train, dv = prepare_x(df_full_train)

model = model_training(x_full_train, y_full_train)



with open ("model.bin", "wb") as f_out:
    pickle.dump((dv, model), f_out)


