from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from ClassifierWrapper import ClassifierWrapper


def get_classifier(method, n_estimators, max_features=None, learning_rate=None, max_depth=None, random_state=None, subsample=None, colsample_bytree=None, min_cases_for_training=30):

    if method == "rf":
        return ClassifierWrapper(
            cls=RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=random_state),
            min_cases_for_training=min_cases_for_training)
               
    elif method == "xgb":
        return ClassifierWrapper(
            cls=xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                                     max_depth=max_depth, colsample_bytree=colsample_bytree, n_jobs=2),
            min_cases_for_training=min_cases_for_training)

    else:
        print("Invalid classifier type")
        return None
