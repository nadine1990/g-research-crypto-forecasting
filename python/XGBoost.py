import xgboost as xgb


def get_Xy_and_model_for_asset(df_train, asset_id, features):
    df = df_train[df_train["Asset_ID"] == asset_id]

    # TODO: Try different features here!
    df_proc = features
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")

    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=11,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        missing=-999,
        random_state=2020,
        # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
    )
    model.fit(X, y)

    return X, y, model