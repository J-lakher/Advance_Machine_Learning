import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load the dataset
file_path = 'climate_change_data.csv'  # Ensure this matches your dataset's filename
df = pd.read_csv(file_path)

# Display initial data information
print("Dataset Head:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Drop rows with missing values
df = df.dropna()

# Step 3: Extract 'Year' from the 'Date' column
df['Year'] = pd.to_datetime(df['Date']).dt.year

# Step 4: Create binary classification target for 'Sea Level Rise'
df['Sea_Level_Classification'] = (df['Sea Level Rise'] > 0).astype(int)

# Step 5: Define features and targets
features = ['Year', 'CO2 Emissions','Wind Speed', 'Humidity']  # Features
target_regression = 'Temperature'  # Regression target
target_classification = 'Sea_Level_Classification'  # Classification target

# Split the data into training and testing sets
X = df[features]
y_regression = df[target_regression]  # Temperature for regression
y_classification = df[target_classification]  # Sea Level Rise for classification

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Step 6: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Save the preprocessed data and scaler
print("Saving preprocessed data and scaler...")
joblib.dump(scaler, 'scaler.pkl')

# Save the training data for regression
pd.DataFrame(X_train_scaled, columns=features).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train_reg).to_csv('y_train_reg.csv', index=False)

# Save the training data for classification
pd.DataFrame(y_train_class).to_csv('y_train_class.csv', index=False)

print("Data preprocessing complete. Files saved:")
print(" - scaler.pkl")
print(" - X_train.csv")
print(" - y_train_reg.csv")
print(" - y_train_class.csv")


