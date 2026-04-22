
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib   # <-- Missing import
# Sample training data: [hours studied], [pass/fail]
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 0 = fail, 1 = pass

model = LogisticRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    joblib.dump(model, f)

with open("model.pkl", "rb") as f:
    model = joblib.load(f)
print(model)