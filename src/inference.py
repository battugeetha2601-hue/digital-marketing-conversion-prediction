import pandas as pd

def score_users(model, X):
    pred = model.predict_proba(X)[:,1]
    return pd.Series(pred, name="pred_prob")