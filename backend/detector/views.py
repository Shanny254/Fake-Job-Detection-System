import joblib
from django.http import JsonResponse
from pathlib import Path

# Resolve project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load model artifacts once (not on every request)
model = joblib.load(BASE_DIR / "Model" / "fraud_model.joblib")
vectorizer = joblib.load(BASE_DIR / "Model" / "tfidf_vectorizer.joblib")


def predict_fraud(request):
    """
    API endpoint that predicts whether a job posting is fraudulent.
    Usage:
    /api/predict/?text=some+job+description
    """
    text = request.GET.get("text", "").strip()

    if not text:
        return JsonResponse(
            {"error": "No text provided"},
            status=400
        )

    X = vectorizer.transform([text])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return JsonResponse({
        "fraudulent": prediction,
        "fraud_probability": round(probability, 3)
    })
