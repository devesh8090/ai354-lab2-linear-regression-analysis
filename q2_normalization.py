import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Load Data
df = pd.read_csv('Salary_dataset.csv')
X = df[['YearsExperience']]
y = df['Salary']


# Fixed 80:20 Split (State 42 ensures the data points are the same for all 3 tests)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("--- QUESTION 2: Normalization Impact ---")


# 1. No Normalization (Raw Data)
model = LinearRegression()
model.fit(X_train, y_train)
mse_none = mean_squared_error(y_test, model.predict(X_test))
print(f"MSE (None): {mse_none:.2f}")


# 2. Min-Max Scaling (Squeezes data between 0 and 1)
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)


model_mm = LinearRegression()
model_mm.fit(X_train_mm, y_train)
mse_mm = mean_squared_error(y_test, model_mm.predict(X_test_mm))
print(f"MSE (MinMax): {mse_mm:.2f}")


# 3. Standard Scaling (Centers data around 0)
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)


model_std = LinearRegression()
model_std.fit(X_train_std, y_train)
mse_std = mean_squared_error(y_test, model_std.predict(X_test_std))
print(f"MSE (Standard): {mse_std:.2f}")


# Create a clean, mathematical scatter plot with regression line
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color='black', marker='o', s=20, label='Data Points')
plt.plot(X, model.predict(X), color='black', linewidth=1, label='Regression Line')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Linear Regression Fit')
plt.legend()
plt.tight_layout()
plt.savefig('q2_graph.png')
print("Graph saved as q2_graph.png")

