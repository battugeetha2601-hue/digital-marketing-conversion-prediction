from sklearn.metrics import roc_auc_score

def evaluate(model, X, y):
    pred = model.predict_proba(X)[:,1]
    return roc_auc_score(y, pred)