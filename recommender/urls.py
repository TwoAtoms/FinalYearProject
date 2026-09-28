from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_view, name="login"),

    path("login/", views.login_view, name="login"),

    path("register/", views.register, name="register"),

    path("logout/", views.logout_view, name="logout"),

    path("home/", views.home, name="home"),

    path(
        "profile/",
        views.student_profile,
        name="student_profile"
    ),

    path(
        "course/<int:course_id>/",
        views.course_detail,
        name="course_detail"
    ),

    path(
        "course/<int:course_id>/like/",
        views.like_course,
        name="like_course"
    ),

    path(
        "liked-courses/",
        views.liked_courses,
        name="liked_courses"
    ),
]
