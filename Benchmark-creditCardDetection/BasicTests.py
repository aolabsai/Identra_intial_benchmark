import pandas as pd
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# -----------------------------
# 1) LOAD & SHUFFLE THE DATA
# -----------------------------
# Make sure to set this path to wherever you've got the downloaded CSV
path = kagglehub.dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")

# Load the dataset
df = pd.read_csv(path+ "/creditcard_2023.csv")

# Shuffle the entire DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# 2) SPLIT INTO TRAIN / TEST
# -----------------------------
# Assume exactly 1000 rows for consistency with your code; if you have more, adjust accordingly
Number_trials = 1000
df = df.iloc[:Number_trials]  # (optional) limit to first 1000 rows

# 80% train / 20% test
train_df, test_df = train_test_split(df, train_size=0.8, stratify=df["Class"], random_state=42)

# -----------------------------
# 3) FEATURE PREPROCESSING
# -----------------------------
# In your code you took V1–V28 as already “anonymized features” (i.e. they’re essentially PCA‐like),
# so we’ll leave those alone. We only need to scale “Amount” before feeding into LR / RF.

scaler = MinMaxScaler()
train_amount = train_df[["Amount"]].values
test_amount = test_df[["Amount"]].values

scaler.fit(train_amount)
train_df["ScaledAmount"] = scaler.transform(train_amount)
test_df["ScaledAmount"] = scaler.transform(test_amount)

# Now, build X and y for our baselines:
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["ScaledAmount"]
X_train = train_df[FEATURE_COLUMNS].values
y_train = train_df["Class"].values

X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df["Class"].values

# -----------------------------
# 4) TRAIN “STANDARD” MODELS
# -----------------------------

# 4a) Logistic Regression
lr = LogisticRegression(solver="lbfgs", max_iter=100, class_weight="balanced", random_state=42)
lr.fit(X_train, y_train)

# 4b) Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight="balanced",
    random_state=42,
)
rf.fit(X_train, y_train)

# You can add more models here (e.g. GradientBoostingClassifier, XGBoost, etc.)

# -----------------------------
# 5) EVALUATE ON TEST SET
# -----------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Some classifiers don’t implement predict_proba; in that case, skip ROC AUC
        pass

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    print(f"\n===== {name} RESULTS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if y_prob is not None:
        print(f"ROC AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


evaluate_model("Logistic Regression", lr, X_test, y_test)
evaluate_model("Random Forest", rf, X_test, y_test)

# -----------------------------
# 6) OPTIONAL: OUTPUT A COMPARISON TABLE
# -----------------------------
# If you want a concise table of metrics, you could gather them in a DataFrame:
from sklearn.metrics import roc_auc_score

models = {"Logistic Regression": lr, "Random Forest": rf}
results = []

for name, mdl in models.items():
    y_pred = mdl.predict(X_test)
    try:
        y_prob = mdl.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    results.append(
        {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC AUC": roc_auc,
        }
    )

results_df = pd.DataFrame(results).set_index("Model")
print("\n====== SUMMARY TABLE ======\n")
print(results_df.round(4))
