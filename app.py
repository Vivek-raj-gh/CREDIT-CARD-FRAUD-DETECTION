import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.decomposition import PCA

# Load Dataset
df = pd.read_csv("creditcard.csv")
df.head()

# Exploratory Data Analysis (EDA)
class_counts = df['Class'].value_counts()

plt.bar(['Normal', 'Fraud'], class_counts)
plt.title("Class Distribution")
plt.show()

print("Fraud Percentage:", class_counts[1] / len(df) * 100)

plt.hist(df['Amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

# Robust Scaling
scaler = RobustScaler()

df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df = df.drop(columns=['Amount'])

# Unsupervised Model Building
normal_df = df[df['Class'] == 0]
fraud_df = df[df['Class'] == 1]

X_train = normal_df.drop(columns=['Class'])
X_test = df.drop(columns=['Class'])
y_test = df['Class']

# Algorithm 1: Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

iso_forest.fit(X_train)

iso_scores = iso_forest.decision_function(X_test)
iso_preds = iso_forest.predict(X_test)

# Convert to binary
iso_preds = [1 if x == -1 else 0 for x in iso_preds]

# Algorithm 2: Local Outlier Factor (LOF)
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.01,
    novelty=True
)

lof.fit(X_train)

lof_preds = lof.predict(X_test)
lof_preds = [1 if x == -1 else 0 for x in lof_preds]

# Evaluation
print("Isolation Forest Report")
print(confusion_matrix(y_test, iso_preds))
print(classification_report(y_test, iso_preds))

print("LOF Report")
print(confusion_matrix(y_test, lof_preds))
print(classification_report(y_test, lof_preds))

# Threshold Tuning
precision, recall, thresholds = precision_recall_curve(y_test, -iso_scores)

f1_scores = 2 * (precision * recall) / (precision + recall)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("Best Threshold:", best_threshold)
print("Best F1:", f1_scores[best_index])

# PCA Visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

plt.scatter(X_pca[:,0], X_pca[:,1], c=iso_preds, cmap='coolwarm')
plt.title("Isolation Forest Anomalies (PCA Projection)")
plt.show()
