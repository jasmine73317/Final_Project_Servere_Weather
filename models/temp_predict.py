import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math
import joblib

df = pd.read_csv("classifier_dataset.csv")

df[f"hr-predict-Temp"] = df["Temperature"].shift(-26, axis=0)

for x in np.arange(1, 109):
    shift_distance = x*26
    df[f"{x}-hrs-ago-Temp"] = df["Temperature"].shift(shift_distance, axis=0)
    df[f"{x}-hrs-ago-Humidity"] = df["Humidity"].shift(shift_distance, axis=0)
    df[f"{x}-hrs-ago-WNDSPD"] = df["Wind_Speed"].shift(shift_distance, axis=0)
    df[f"{x}-hrs-ago-WNDDIR"] = df["Wind_Direction"].shift(shift_distance, axis=0)
    df[f"{x}-hrs-ago-Pressure"] = df["Pressure"].shift(shift_distance, axis=0)

without_previous_data = 108 * 26
without_future_data = -1 * 26

df = df.iloc[without_previous_data:without_future_data,:]

# Assign features

features = ['Timestamp', 'Temperature', 'Humidity', 'Wind_Speed',
    'Wind_Direction', 'Pressure', 'Year', 'Day sin', 'Day cos', 'Year sin', 'Year cos',
    'Latitude', 'Longitude']

for x in np.arange(1, 109):
    features.append(f"{x}-hrs-ago-Temp")
    features.append(f"{x}-hrs-ago-Humidity")
    features.append(f"{x}-hrs-ago-WNDSPD")
    features.append(f"{x}-hrs-ago-WNDDIR")
    features.append(f"{x}-hrs-ago-Pressure")

selected_features = df[features]
y = df[['hr-predict-Temp']]

y = y.values.reshape(-1,1)

# Split data

length = math.floor(len(df)*0.8)
X_train = selected_features[:length]
y_train = y[:length]
X_test = selected_features[length:]
y_test = y[length:]

# Scale data

X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Fit model

model = LinearRegression()

model.fit(X_train_scaled, y_train_scaled)

# Print Scores

print(f"Training Data Score: {model.score(X_train_scaled, y_train_scaled)}")
print(f"Testing Data Score: {model.score(X_test_scaled, y_test_scaled)}")
predictions = model.predict(X_test_scaled)
print(f"Mean Absolute Error: {mean_absolute_error(y_test_scaled, predictions)}")


# Save model

filename = 'hr_temp_predict_model.h5'
joblib.dump(model, filename)