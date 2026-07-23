# Import Libraries
import pandas as pd

# Load Dataset
df = pd.read_csv("bank.csv")

# Show Dataset
print(df.head())

# Dataset Information
print(df.info())

# Column Names
print(df.columns)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode Target Column
df["deposit"] = df["deposit"].map({"no": 0, "yes": 1})

# Convert Categorical Columns
df = pd.get_dummies(df, drop_first=True)

# Display Updated Data
print("\nUpdated Dataset:")
print(df.head())

from sklearn.model_selection import train_test_split

# Features and Target
X = df.drop("deposit", axis=1)
y = df["deposit"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Display Shapes
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

from sklearn.linear_model import LogisticRegression

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Predict Test Data
y_pred = model.predict(X_test)

print("\nFirst 10 Actual Values:")
print(y_test.head(10).values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])

# Predict Test Data
y_pred = model.predict(X_test)

print("\nFirst 10 Actual Values:")
print(y_test.head(10).values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])

import joblib

# Save Model
joblib.dump(model, "bank_model.pkl")

# Save Column Names
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nModel Saved Successfully!")

# Load Saved Model
loaded_model = joblib.load("bank_model.pkl")
loaded_columns = joblib.load("columns.pkl")

# Sample Input
sample = X.iloc[[0]]

# Predict
prediction = loaded_model.predict(sample)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Customer Will Subscribe: Yes")
else:
    print("Customer Will Subscribe: No")