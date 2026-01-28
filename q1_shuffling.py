import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load Data
df = pd.read_csv('Salary_dataset.csv')
X = df[['YearsExperience']]
y = df['Salary']


print("--- QUESTION 1: Shuffling Impact ---")


mse_scores = []
# We use 5 specific numbers (seeds) to make sure we shuffle differently each time
shuffle_seeds = [10, 20, 30, 40, 50]


for i, seed in enumerate(shuffle_seeds):
    # Split data 70% for training, 30% for testing
    # random_state ensures we get a specific shuffle for this loop iteration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print(f"Run {i+1} (Seed {seed}): MSE = {mse:.2f}")


# Create a clean, mathematical bar chart
plt.figure(figsize=(6, 4))
plt.bar(['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5'], mse_scores, color='gray', edgecolor='black')
plt.ylabel('Mean Squared Error')
plt.title('MSE Variation across 5 Shuffles')
plt.tight_layout()
plt.savefig('q1_graph.png')
print("Graph saved as q1_graph.png"