from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.features import create_features
from src.train import train_model
from src.evaluate import evaluate
from src.config import TARGET, DROP_COLS, RANDOM_STATE

df = load_data("data/raw/digital_marketing_campaign_dataset.csv")

df = preprocess_data(df)
df = create_features(df)

# Drop the target, identified leakage columns, and the raw string columns from config
cols_to_drop = [TARGET, "ConversionRate"] + DROP_COLS
X = df.drop(columns=cols_to_drop)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

model = train_model(X_train, y_train)

score = evaluate(model, X_test, y_test)

print("ROC-AUC:", round(score, 4))