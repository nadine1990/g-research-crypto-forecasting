import pandas as pd
import numpy as np

from python.LGBMRegressor import get_Xy_and_model_for_asset
from python.feature_extraction import get_features
from python.preprocess import preprocess

TRAIN_CSV = '../input/train.csv'
ASSET_DETAILS_CSV = '../input/asset_details.csv'

training_set = preprocess(pd.read_csv(TRAIN_CSV))
asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")

# TODO: Try different features here!
features = get_features(training_set)

Xs = {}
ys = {}
models = {}

# Training model
for asset_id, asset_name in zip(asset_details['Asset_ID'], asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(training_set, asset_id, features)
    Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model

# The competition performance metric is weighted correlation.
# However, for now we will use simple correlation to evaluate the two baseline models built.
# print('Test score for LR baseline: BTC', f"{np.corrcoef(y_pred_lr_btc, y_btc_test)[0,1]:.2f}",
#                                ', ETH', f"{np.corrcoef(y_pred_lr_eth, y_eth_test)[0,1]:.2f}")


