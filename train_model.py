import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('predictive_maintenance.csv')

# Data preprocessing
# Handle missing values
data = data.dropna()

# -----Target Prediction Model-----
# Define features and target for target prediction
X_target = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y_target = data['Target']

# Split data into training and validation sets
X_train_target, X_val_target, y_train_target, y_val_target = train_test_split(X_target, y_target, test_size=0.2, random_state=42)

# Instantiate and train the target prediction model
model_target = RandomForestRegressor()
model_target.fit(X_train_target, y_train_target)

# Save the trained model
joblib.dump(model_target, 'model_target.pkl')

# Predictions and evaluation for target prediction
y_pred_target = model_target.predict(X_val_target)
mse_target = mean_squared_error(y_val_target, y_pred_target)
print(f'Target Prediction Mean Squared Error: {mse_target}')

# -----Failure Type Prediction Model-----
# Define features and target for failure type prediction
X_failure = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y_failure = data['Failure Type']

# Encode categorical target variable
le_failure_type = LabelEncoder()
y_failure_encoded = le_failure_type.fit_transform(y_failure)

# Save the label encoder
joblib.dump(le_failure_type, 'label_encoder.pkl')

# Split data into training and validation sets
X_train_failure, X_val_failure, y_train_failure, y_val_failure = train_test_split(X_failure, y_failure_encoded, test_size=0.2, random_state=42)

# Instantiate and train the failure type prediction model
model_failure_type = RandomForestClassifier()
model_failure_type.fit(X_train_failure, y_train_failure)

# Save the trained model
joblib.dump(model_failure_type, 'model_failure_type.pkl')

# Predictions and evaluation for failure type prediction
y_pred_failure = model_failure_type.predict(X_val_failure)
accuracy_failure = accuracy_score(y_val_failure, y_pred_failure)
print(f'Failure Type Prediction Accuracy: {accuracy_failure}')
print(classification_report(y_val_failure, y_pred_failure, target_names=le_failure_type.classes_))

print("Models trained and saved successfully.")
