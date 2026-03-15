import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/housing.csv")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Convert categorical column
df = pd.get_dummies(df, columns=["ocean_proximity"])

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)
dataset_size = len(df)

print("Dataset size:", dataset_size)
print("RMSE:", rmse)
print("R2:", r2)

with open("metrics.txt", "w") as f:
    f.write(f"{dataset_size},{rmse},{r2}")