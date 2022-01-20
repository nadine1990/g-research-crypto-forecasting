from python.feature_extraction import get_features
from lightgbm import LGBMRegressor


def get_Xy_and_model_for_asset(df_train, asset_id, features):
    df = df_train[df_train["Asset_ID"] == asset_id]

    df_proc = features
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")

    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]

    # TODO: Try different models here!
    model = LGBMRegressor(n_estimators=10)
    model.fit(X, y)
    return X, y, model