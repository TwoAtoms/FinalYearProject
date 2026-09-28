from django.db import models
from django.contrib.auth.models import User


class UserInteraction(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE
    )

    course_id = models.IntegerField()

    rating = models.IntegerField(
        null=True,
        blank=True
    )

    viewed = models.BooleanField(
        default=False
    )

    liked = models.BooleanField(
        default=False
    )

    timestamp = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "course_id"],
                name="unique_user_course"
            )
        ]

    def __str__(self):
        return f"{self.user.username} - Course {self.course_id}"


class StudentProfile(models.Model):

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="student_profile"
    )

    programme = models.CharField(
        max_length=150,
        blank=True
    )

    year_of_study = models.IntegerField(
        null=True,
        blank=True
    )

    completed_courses = models.JSONField(
        default=list,
        blank=True
    )

    def __str__(self):
        return self.user.username
