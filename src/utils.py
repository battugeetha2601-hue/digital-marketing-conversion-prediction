import pandas as pd

def feature_importance(model, cols):
    imp = pd.Series(model.feature_importances_, index=cols)
    return imp.sort_values(ascending=False)