import pandas as pd

def make_predictions(config):
    df_sales = pd.read_csv(config["data"]["sales"])

    df_sales["prediction"] = df_sales.groupby("item_id")["sales"].shift(1)

    df_sales = df_sales[df_sales["dates"] >= config["start_test"]].reset_index(drop=True)

    return df_sales[["dates", "item_id", "prediction"]]
