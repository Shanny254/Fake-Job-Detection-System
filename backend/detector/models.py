from django.db import models
from django.contrib.auth.models import User


class JobAnalysis(models.Model):

    job_text = models.TextField()

    fraudulent = models.IntegerField(
        help_text="1 = fraudulent, 0 = legitimate"
    )

    probability = models.FloatField(
        help_text="Fraud probability score"
    )

    submitted_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="job_analyses"
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"Analysis {self.id} - Fraud: {self.fraudulent}"