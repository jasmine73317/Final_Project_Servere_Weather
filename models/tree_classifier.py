import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn import tree
import joblib

df = pd.read_csv("classifier_dataset.csv")

selected_features = df[['Timestamp', 'Temperature', 'Humidity', 'Wind_Speed',
    'Wind_Direction', 'Pressure', 'Year', 'Day sin', 'Day cos', 'Year sin',
    'Year cos', 'Latitude', 'Longitude']]
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

model = tree.DecisionTreeClassifier()
model.fit(X_train_scaled, y_train_categorical)

print(f"Training Data Score: {model.score(X_train_scaled, y_train_categorical)}")
print(f"Testing Data Score: {model.score(X_test_scaled, y_test_categorical)}")

filename = 'decision_tree_classifier_model.h5'
joblib.dump(model, filename)