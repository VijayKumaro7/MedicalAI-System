# ml_models/risk_classifier.py
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class DiseaseRiskClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ))
        ])
        self.disease_labels = {
            0: 'Healthy',
            1: 'Diabetes Risk',
            2: 'Heart Disease Risk',
            3: 'Hypertension Risk'
        }

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        score = self.pipeline.score(X_test, y_test)

        print(f"✅ Model Accuracy: {score:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(self.disease_labels.values())))

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.pipeline, 'models/risk_classifier.pkl')
        print("Model saved to models/risk_classifier.pkl")
        return score

    def load(self, path: str = 'models/risk_classifier.pkl'):
        self.pipeline = joblib.load(path)
        return self

    def predict_risk(self, patient_data: dict) -> dict:
        """
        Predict disease risk from structured patient data.

        Args:
            patient_data: dict with keys like age, bmi, glucose,
                          blood_pressure, cholesterol, smoking, family_history

        Returns:
            dict with diagnosis, confidence, and risk_scores
        """
        df = pd.DataFrame([patient_data])
        proba = self.pipeline.predict_proba(df)[0]
        predicted_class = self.pipeline.predict(df)[0]

        return {
            "diagnosis": self.disease_labels[predicted_class],
            "confidence": round(float(max(proba)) * 100, 2),
            "risk_scores": {
                self.disease_labels[i]: round(float(p) * 100, 2)
                for i, p in enumerate(proba)
            }
        }


def generate_sample_data(n_samples: int = 1000):
    """Generate synthetic patient data for demonstration."""
    import numpy as np
    np.random.seed(42)

    data = {
        'age': np.random.randint(18, 80, n_samples),
        'bmi': np.round(np.random.uniform(18.5, 40.0, n_samples), 1),
        'glucose': np.random.randint(70, 200, n_samples),
        'blood_pressure': np.random.randint(60, 120, n_samples),
        'cholesterol': np.random.randint(150, 300, n_samples),
        'smoking': np.random.randint(0, 2, n_samples),
        'family_history': np.random.randint(0, 2, n_samples),
    }
    df = pd.DataFrame(data)

    # Simple rule-based labels for demo
    conditions = [
        (df['glucose'] > 140) & (df['bmi'] > 28),
        (df['cholesterol'] > 240) | (df['blood_pressure'] > 100),
        (df['blood_pressure'] > 95) & (df['age'] > 45),
    ]
    labels = np.zeros(n_samples, dtype=int)
    for i, cond in enumerate(conditions, 1):
        labels[cond.values] = i

    return df, pd.Series(labels)


if __name__ == "__main__":
    X, y = generate_sample_data(2000)
    clf = DiseaseRiskClassifier()
    clf.train(X, y)

    # Test prediction
    sample_patient = {
        'age': 52, 'bmi': 28.4, 'glucose': 145,
        'blood_pressure': 92, 'cholesterol': 220,
        'smoking': 1, 'family_history': 1
    }
    result = clf.predict_risk(sample_patient)
    print("\nSample Prediction:", result)
