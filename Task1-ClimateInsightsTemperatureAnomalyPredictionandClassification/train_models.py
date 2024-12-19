import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
# Step 1: Load preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('X_train.csv')  # Scaled features
y_train_reg = pd.read_csv('y_train_reg.csv').values.ravel()  # Regression target
y_train_class = pd.read_csv('y_train_class.csv').values.ravel()  # Classification target

# Step 2: Train Regression Model
print("\nTraining Regression Model...")
regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
regression_model.fit(X_train, y_train_reg)

# Step 3: Train Classification Model
print("\nTraining Classification Model...")
classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
classification_model.fit(X_train, y_train_class)

# Step 4: Evaluate Models on Training Data
print("\nEvaluating Models on Training Data...")

# Regression Model Evaluation
y_pred_reg = regression_model.predict(X_train)
reg_mse = mean_squared_error(y_train_reg, y_pred_reg)
print(f"Regression Model - Mean Squared Error (Training): {reg_mse:.2f}")

# Classification Model Evaluation
y_pred_class = classification_model.predict(X_train)
class_accuracy = accuracy_score(y_train_class, y_pred_class)
print(f"Classification Model - Accuracy (Training): {class_accuracy:.2f}")

# Step 5: Save the Models
print("\nSaving Models...")
joblib.dump(regression_model, 'model_regression.pkl')
joblib.dump(classification_model, 'model_classification.pkl')
print("Models saved successfully:")
print(" - model_regression.pkl")
print(" - model_classification.pkl")

print("\nModel training complete.")
