import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# --------------------------------------------------
# 1. Load Dataset (robust encoding)
# --------------------------------------------------
for enc in ("utf-8", "latin-1", "cp1252"):
    try:
        df = pd.read_csv("dataset/recruitment-dataset.csv", encoding=enc)
        print(f"Loaded dataset using {enc}")
        break
    except UnicodeDecodeError:
        continue

df.drop(columns=["Unnamed: 0"], inplace=True)
df.dropna(inplace=True)

# --------------------------------------------------
# 2. Feature Selection (from EDA)
# --------------------------------------------------
DROP_COLS = ["YearsCode", "Age", "MentalHealth", "Accessibility", "Gender"]
df.drop(columns=DROP_COLS, inplace=True)

TARGET = "Employed"
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# --------------------------------------------------
# 3. Train / Validation / Test Split (60/20/20)
# --------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# --------------------------------------------------
# 4. Column Groups
# --------------------------------------------------
TEXT_COL = "HaveWorkedWith"

NUM_COLS = ["Employment", "YearsCodePro", "PreviousSalary", "ComputerSkills"]

CAT_COLS = [
    "EdLevel",
    "MainBranch",
    "Country"
]

# --------------------------------------------------
# 5. Preprocessing Pipeline
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("text", TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=5), TEXT_COL)
    ]
)

# --------------------------------------------------
# 6. Model Pipeline
# --------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=2.0,
                solver="liblinear",
                random_state=42
            ),
        ),
    ]
)

# --------------------------------------------------
# 7. Train Model
# --------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------
# 8. Evaluate
# --------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nMODEL PERFORMANCE")
print("-----------------")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 9. Save Model
# --------------------------------------------------
joblib.dump(model, "hr_recruitment_model.pkl")
print("\nModel saved as hr_recruitment_model.pkl")