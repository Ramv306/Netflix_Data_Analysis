import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("netflix_cleaned.csv")

# Step 1: features
y = df["type_code"]
num = df[["release_year", "year_added", "genre_count", "years_to_add"]]
cat = pd.get_dummies(df[["rating"]])
X = pd.concat([num, cat], axis=1).fillna(0)

# Step 2: split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: train
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
for name, m in models.items():
    m.fit(X_train, y_train)

# Step 4: evaluate
for name, m in models.items():
    pred = m.predict(X_test)
    print("\n", name)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

# Step 5: compare accuracy
scores = {n: accuracy_score(y_test, m.predict(X_test)) for n, m in models.items()}
print(scores)

pd.Series(scores).plot(kind="bar")
plt.ylim(0, 1)
plt.title("Model accuracy comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.show()