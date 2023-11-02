import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv("sinking_incident_data.csv")

# Prepare the data
X = data[["age", "gender", "socio_economic_status", "swimming_ability"]]
y = data["survival"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Deploy the model
def predict_survival(age, gender, socio_economic_status, swimming_ability):
  """Predicts whether a person is likely to survive a sinking incident.

  Args:
    age: The person's age.
    gender: The person's gender.
    socio_economic_status: The person's socio-economic status.
    swimming_ability: The person's swimming ability.

  Returns:
    A boolean value indicating whether the person is likely to survive a sinking incident.
  """

  features = [[age, gender, socio_economic_status, swimming_ability]]
  predictions = model.predict(features)
  return predictions[0]

# Example usage:

age = 30
gender = "female"
socio_economic_status = "high"
swimming_ability = "good"

prediction = predict_survival(age, gender, socio_economic_status, swimming_ability)

if prediction:
  print("The person is likely to survive a sinking incident.")
else:
  print("The person is unlikely to survive a sinking incident.")
