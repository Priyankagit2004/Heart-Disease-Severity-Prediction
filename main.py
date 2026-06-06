import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)
from scipy import stats
import numpy as np

# Read CSV from local path 
df = pd.read_csv("C:/Users/sable/Downloads/heart.csv")

# Check for missing values
print("Missing values in original dataset:\n", df.isnull().sum())
# Drop rows with missing values if any
df.dropna(inplace=True)
print("\nAfter dropping missing values:\n", df.isnull().sum())

# Adding a row called severity which will be our target
def assign_severity(row):
    score = 0
    if row['thalach'] < 120:
        score += 1
    if row['oldpeak'] > 2:
        score += 1
    if row['cp'] in [0, 3]:
        score += 1
    if row['ca'] > 0:
        score += 1

    # Map total score to severity levels 1-4
    if score == 0:
        return 0
    elif score == 1:
        return 1
    elif score == 2:
        return 2
    elif score == 3:
        return 3
    else:
        return 4

df['severity'] = df.apply(assign_severity, axis=1)

#Displaying the data and its information
print(df.head())
print(df.info())
print(df.shape)

columns_to_check = ['chol', 'trestbps', 'thalach', 'oldpeak']

for column in columns_to_check:
    z_scores = np.abs(stats.zscore(df[column]))
    df = df[z_scores < 4]

print(df.shape)

# Backup severity before scaling
severity = df['severity']

# Visualize data target for severity
plt.scatter(df['age'], df['severity'], color='blue', label='age')
plt.scatter(df['trestbps'], df['severity'], color='red', label='trestbps')
plt.scatter(df['chol'], df['severity'], color='green', label='chol')
plt.scatter(df['thalach'], df['severity'], color='orange', label='thalach')
plt.scatter(df['oldpeak'], df['severity'], color='purple', label='oldpeak')
plt.scatter(df['ca'], df['severity'], color='brown', label='ca')
plt.scatter(df['cp'], df['severity'], color='pink', label='cp')
plt.scatter(df['thal'], df['severity'], color='gray', label='thal')
plt.scatter(df['sex'], df['severity'], color='black', label='sex')
plt.scatter(df['slope'], df['severity'], color='yellow', label='slope')
plt.scatter(df['exang'], df['severity'], color='teal', label='exang')
plt.scatter(df['fbs'], df['severity'], color='olive', label='fbs')
plt.scatter(df['restecg'], df['severity'], color='lime', label='restecg')
plt.title('Severity vs Features')
plt.xlabel('Features')
plt.ylabel('Severity')
plt.legend()
plt.show()

# Random Forest for Feature Importance
x_rf = df.drop(columns=["target", "severity"])
y_rf = df["severity"]
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_rf, y_rf)

# Get feature importance scores
feature_importances = pd.Series(rf_model.feature_importances_, index=x_rf.columns)
sorted_features = feature_importances.sort_values(ascending=False)
top_features = sorted_features[:5].index.tolist()

# Sort and plot feature importance
plt.figure(figsize=(10, 5))
feature_importances.sort_values(ascending=False).plot(kind="bar", color="teal")
plt.title("Feature Importance using Random Forest")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.show()

# Print top features
print("\nTop 5 Important Features:\n", sorted_features[:5])

#Select features
features = ['thalach', 'oldpeak', 'cp', 'ca']
x = df[features]
y = df['severity']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("x train :\n", x_train, "\nx test :\n" , x_test, "\ny train :\n", y_train, "\ny test :\n", y_test)

#Scalling the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#-----1. Gaussian Naive Bayes for Severity Prediction-----

print("\n\n---1. Gaussian Naive Bayes---\n\n")

#Training the model
gnb = GaussianNB()
gnb.fit(x_train, y_train)

#Predicting the test values
y_pred = gnb.predict(x_test)
print(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#Calculating the test metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


#-----2. Support Vector Machine (SVM) for Severity Prediction-----

print("\n\n---2. Support Vector Machines---\n\n")

# Support Vector Machine (SVM) for Severity Prediction
svm_model = SVC(kernel='linear', C=0.1, gamma='auto', random_state=42)
svm_model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(x_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.show()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

#3. Severity Prediction using Multinomial logistic regression

print("\n\n---3. Multinomail Logistic Regression---\n\n")

# Train the Multinomial Logistic Regression model using existing scaled data
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Optional: Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Multinomial Logistic Regression")
plt.show()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

#4. Random Forest for Severity Prediction

print("\n\n---4. Random Forest---\n\n")

clf = RandomForestClassifier(
    n_estimators=10,
    max_depth=4,
    max_features='log2',    
    min_samples_split=7,
    random_state=42
)

clf.fit(x_train, y_train)

# Predictions
y_pred = clf.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)