import csv
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import JobAnalysis
import joblib
import traceback


# Resolve project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load ML model and vectorizer
try:
    model_path = BASE_DIR / "Model" / "fraud_model.joblib"
    vectorizer_path = BASE_DIR / "Model" / "tfidf_vectorizer.joblib"
    
    print(f"Loading model from: {model_path}")
    print(f"Loading vectorizer from: {vectorizer_path}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("Model and vectorizer loaded successfully!")
    
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    vectorizer = None


def export_predictions():
    """Export all predictions to CSV for Power BI"""
    try:
        export_folder = BASE_DIR / "backend" / "exports"
        export_folder.mkdir(parents=True, exist_ok=True)
        
        file_path = export_folder / "fraud_predictions.csv"
        
        data = JobAnalysis.objects.all()
        
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            
            writer.writerow([
                "ID",
                "Job Text",
                "Fraudulent",
                "Probability",
                "Date",
                "User"
            ])
            
            for item in data:
                writer.writerow([
                    item.id,
                    item.job_text,
                    item.fraudulent,
                    item.probability,
                    item.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    item.submitted_by.username if item.submitted_by else "Anonymous"
                ])
        
        print(f"CSV exported to: {file_path}")
        
    except Exception as e:
        print(f"CSV export failed: {e}")
        traceback.print_exc()


def run_prediction(text):
    """Run ML prediction on job text"""
    if model is None or vectorizer is None:
        raise Exception("Model or vectorizer not loaded. Check if model files exist.")
    
    X = vectorizer.transform([text])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    
    print(f"Prediction: {prediction}, Probability: {probability:.3f}")
    
    return prediction, probability


def predict_fraud(request):
    """API endpoint for fraud prediction"""
    text = request.GET.get("text", "").strip()

    if not text:
        return JsonResponse({"error": "No text provided"}, status=400)

    try:
        existing = JobAnalysis.objects.filter(job_text=text).first()
        
        if existing:
            return JsonResponse({
                "fraudulent": existing.fraudulent,
                "fraud_probability": round(existing.probability, 3),
                "note": "Returning cached result - this job was already analyzed"
            })
        
        prediction, probability = run_prediction(text)

        JobAnalysis.objects.create(
            job_text=text,
            fraudulent=prediction,
            probability=probability,
            submitted_by=request.user if request.user.is_authenticated else None
        )

        export_predictions()

        return JsonResponse({
            "fraudulent": prediction,
            "fraud_probability": round(probability, 3)
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def home(request):
    """Main page with job analysis form"""
    result = None
    error = None

    if request.method == "POST":
        text = request.POST.get("job_text", "").strip()

        if not text:
            error = "Please enter a job description."

        elif len(text.split()) < 10:
            error = "Job description is too short to analyze."

        else:
            try:
                print(f"Analyzing job text: {text[:100]}...")
                
                existing = JobAnalysis.objects.filter(job_text=text).first()
                
                if existing:
                    result = {
                        "fraudulent": existing.fraudulent,
                        "fraud_probability": round(existing.probability, 3),
                        "is_cached": True
                    }
                else:
                    prediction, probability = run_prediction(text)

                    JobAnalysis.objects.create(
                        job_text=text,
                        fraudulent=prediction,
                        probability=probability,
                        submitted_by=request.user
                    )

                    export_predictions()

                    result = {
                        "fraudulent": prediction,
                        "fraud_probability": round(probability, 3),
                        "is_cached": False
                    }

            except Exception as e:
                error = f"Analysis failed: {str(e)}"
                print(f"Error in home view: {e}")
                traceback.print_exc()

    # Show only user's own history (unless admin)
    if request.user.is_superuser:
        history = JobAnalysis.objects.order_by("-id")[:5]
    else:
        history = JobAnalysis.objects.filter(submitted_by=request.user).order_by("-id")[:5]

    return render(
        request,
        "detector/index.html",
        {
            "result": result,
            "error": error,
            "history": history
        }
    )


@login_required
def dashboard(request):
    """Personal Dashboard - User's own statistics OR Admin Dashboard"""
    # Redirect admins to admin dashboard
    if request.user.is_superuser:
        return redirect("admin_dashboard")
    
    # Regular users see only their own data
    total = JobAnalysis.objects.filter(submitted_by=request.user).count()
    fake = JobAnalysis.objects.filter(submitted_by=request.user, fraudulent=1).count()
    real = JobAnalysis.objects.filter(submitted_by=request.user, fraudulent=0).count()

    return render(
        request,
        "detector/dashboard.html",
        {
            "total": total,
            "fake": fake,
            "real": real
        }
    )


@login_required
def admin_dashboard(request):
    """Admin Dashboard - System Overview and User Management"""
    # Redirect non-admins
    if not request.user.is_superuser:
        return redirect("dashboard")
    
    # System-wide statistics
    total_jobs = JobAnalysis.objects.count()
    fake_jobs = JobAnalysis.objects.filter(fraudulent=1).count()
    real_jobs = JobAnalysis.objects.filter(fraudulent=0).count()
    
    # User statistics
    total_users = User.objects.count()
    admin_users = User.objects.filter(is_superuser=True).count()
    regular_users = User.objects.filter(is_superuser=False).count()
    
    # Recent activity
    recent_analyses = JobAnalysis.objects.select_related('submitted_by').order_by('-id')[:10]
    
    # All users
    all_users = User.objects.all().order_by('-date_joined')
    
    return render(
        request,
        "detector/admin_dashboard.html",
        {
            "total_jobs": total_jobs,
            "fake_jobs": fake_jobs,
            "real_jobs": real_jobs,
            "total_users": total_users,
            "admin_users": admin_users,
            "regular_users": regular_users,
            "recent_analyses": recent_analyses,
            "all_users": all_users
        }
    )


@login_required
def history(request):
    """History page - User's own analyzed jobs OR All jobs for admin"""
    # Admin sees all jobs
    if request.user.is_superuser:
        jobs = JobAnalysis.objects.select_related('submitted_by').order_by("-id")
    else:
        # Users see only their own jobs
        jobs = JobAnalysis.objects.filter(submitted_by=request.user).order_by("-id")

    return render(
        request,
        "detector/history.html",
        {
            "jobs": jobs
        }
    )


@login_required
def analytics(request):
    """Global Analytics - Platform-wide statistics (everyone sees same data)"""
    # GLOBAL STATS - All users combined
    total_global = JobAnalysis.objects.count()
    fake_global = JobAnalysis.objects.filter(fraudulent=1).count()
    real_global = JobAnalysis.objects.filter(fraudulent=0).count()
    
    # Total users who have analyzed jobs
    total_users = JobAnalysis.objects.values('submitted_by').distinct().count()
    
    # Calculate fraud rate
    fraud_rate = round((fake_global / total_global * 100) if total_global > 0 else 0, 1)

    return render(
        request,
        "detector/analytics.html",
        {
            "fake": fake_global,
            "real": real_global,
            "total": total_global,
            "total_users": total_users,
            "fraud_rate": fraud_rate
        }
    )


def login_view(request):
    """Login page"""
    error = None
    
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            # Redirect based on user type
            if user.is_superuser:
                return redirect("admin_dashboard")
            else:
                return redirect("analyze")
        else:
            error = "Invalid username or password"
    
    return render(request, "detector/login.html", {"error": error})


def signup_view(request):
    """Signup page for new users"""
    error = None
    success = None
    
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")
        
        # Validation
        if not username or not email or not password:
            error = "All fields are required"
        elif password != confirm_password:
            error = "Passwords do not match"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        elif User.objects.filter(username=username).exists():
            error = "Username already exists"
        elif User.objects.filter(email=email).exists():
            error = "Email already registered"
        else:
            # Create new user (regular user by default)
            User.objects.create_user(
                username=username,
                email=email,
                password=password,
                is_staff=False,
                is_superuser=False
            )
            success = True
            
    return render(request, "detector/signup.html", {
        "error": error,
        "success": success
    })


def logout_view(request):
    """Logout user"""
    logout(request)
    return redirect("login")