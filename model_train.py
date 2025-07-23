import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
df = pd.read_csv("car_data.csv")

# Print columns to confirm names (debug tip)
print("Columns:", df.columns.tolist())

# Separate features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Drop 'Car_Name' as it is not useful for prediction
X = X.drop("Car_Name", axis=1)

# Define categorical and numerical features
categorical_features = ["Fuel_Type", "Seller_Type", "Transmission"]
numerical_features = ["Year", "Present_Price", "Kms_Driven", "Owner"]

# Create transformers
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ]
)

# Create pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Save the model using pickle
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as model.pkl")
