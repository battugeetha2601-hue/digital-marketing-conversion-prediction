from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=150,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model