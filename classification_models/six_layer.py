import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import math
import joblib

df = pd.read_csv("classifier_dataset.csv")

selected_features = df[['Timestamp', 'Temperature', 'Humidity', 'Wind_Speed',
    'Wind_Direction', 'Pressure', 'Year', 'Day sin', 'Day cos', 'Year sin', 'Year cos',
    'Latitude', 'Longitude']]
y = df[["Weather_Description"]]

y = y.values.reshape(-1,)

label_encoder = LabelEncoder()
label_encoder.fit(y)
encoded_y = label_encoder.transform(y)

y_categorical = to_categorical(encoded_y)

X_train, X_test, y_train_categorical, y_test_categorical = train_test_split(selected_features, y_categorical, test_size=0.20)

X_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

inputs = selected_features.shape[1]
classes = len(df["Weather_Description"].unique())

model = Sequential()

model.add(Dense(units=150, activation='relu', input_dim=inputs))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=600, activation='relu'))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=100, activation='relu'))

model.add(Dense(units=classes, activation='softmax'))

model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

model.fit(X_train_scaled,
    y_train_categorical,
    epochs=20,
    shuffle=True,
    verbose=2)

print(f"Training Data Score: {model.evaluate(X_train_scaled, y_train_categorical)}")
print(f"Testing Data Score: {model.evaluate(X_test_scaled, y_test_categorical)}")

filename = 'six_layer_model.h5'
model.save(filename)

joblib.dump(X_scaler, "six_layer_scaler.joblib")

joblib.dump(label_encoder, "six_layer_encoder.joblib")