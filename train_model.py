import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the complete dataset
df = pd.read_csv("data/advertising.csv")

print("Original dataset shape:", df.shape)


# 2. Data preprocessing
# Remove the row containing missing values
df = df.dropna()

print("Dataset shape after removing missing values:", df.shape)

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

print("\nDuplicate rows kept in dataset:", df.duplicated().sum())


# 3. Define features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. Build the Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)


# 8. Save the trained model
model_path = "model/sales_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully at:", model_path)