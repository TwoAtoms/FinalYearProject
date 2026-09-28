from django.shortcuts import render, redirect
from django.core.paginator import Paginator
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count

import pandas as pd

from .forms import (
    RegisterForm,
    StudentProfileForm
)

from .models import (
    UserInteraction,
    StudentProfile
)

from .recommendation import (
    get_recommendations,
    get_user_recommendations,
    data,
    similarity_matrix
)


# =========================================================
# REGISTER
# =========================================================

def register(request):

    if request.method == "POST":

        form = RegisterForm(
            request.POST
        )

        if form.is_valid():

            user = form.save(
                commit=False
            )

            user.set_password(
                form.cleaned_data["password"]
            )

            user.save()

            # Create student profile automatically
            StudentProfile.objects.get_or_create(
                user=user
            )

            messages.success(
                request,
                "Your account has been created successfully."
            )

            return redirect("login")

    else:

        form = RegisterForm()

    return render(
        request,
        "register.html",
        {
            "form": form
        }
    )


# =========================================================
# LOGIN
# =========================================================

def login_view(request):

    if request.method == "POST":

        username = request.POST.get(
            "username",
            ""
        )

        password = request.POST.get(
            "password",
            ""
        )

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(
                request,
                user
            )

            return redirect("home")

        messages.error(
            request,
            "Incorrect username or password. Please try again."
        )

    return render(
        request,
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

def logout_view(request):

    logout(request)

    return redirect("login")


# =========================================================
# HOME
# =========================================================

@login_required
def home(request):

    search = request.GET.get(
        "search",
        ""
    ).strip()

    sort = request.GET.get(
        "sort",
        "recommended"
    )

    courses = data.copy()

    # -----------------------------------------------------
    # Rating sorting value
    # -----------------------------------------------------

    courses["rating_sort"] = pd.to_numeric(
        courses["rating"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    if search:

        courses = courses[
            courses["course_name"].str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ]

    # -----------------------------------------------------
    # View counts
    # -----------------------------------------------------

    view_counts = {
        item["course_id"]: item["count"]
        for item in (
            UserInteraction.objects
            .filter(viewed=True)
            .values("course_id")
            .annotate(
                count=Count("id")
            )
        )
    }

    # -----------------------------------------------------
    # Like counts
    # -----------------------------------------------------

    like_counts = {
        item["course_id"]: item["count"]
        for item in (
            UserInteraction.objects
            .filter(liked=True)
            .values("course_id")
            .annotate(
                count=Count("id")
            )
        )
    }

    # -----------------------------------------------------
    # Add counts
    # -----------------------------------------------------

    courses["view_count"] = (
        courses["course_id"]
        .map(view_counts)
        .fillna(0)
        .astype(int)
    )

    courses["like_count"] = (
        courses["course_id"]
        .map(like_counts)
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    if sort == "most_viewed":

        courses = courses.sort_values(
            "view_count",
            ascending=False
        )

    elif sort == "most_liked":

        courses = courses.sort_values(
            "like_count",
            ascending=False
        )

    else:

        sort = "recommended"

        recommendations = (
            get_user_recommendations(
                request.user,
                data,
                similarity_matrix,
                top_n=len(data)
            )
        )

        if recommendations:

            recommendation_ids = [
                item["course_id"]
                for item in recommendations
            ]

            recommendation_order = {
                course_id: index
                for index, course_id
                in enumerate(
                    recommendation_ids
                )
            }

            courses["recommendation_order"] = (
                courses["course_id"].map(
                    recommendation_order
                )
            )

            courses = courses.sort_values(
                "recommendation_order",
                na_position="last"
            )

        else:

            # -------------------------------------------------
            # Fallback for completely new users
            # -------------------------------------------------

            courses = courses.sort_values(
                [
                    "rating_sort",
                    "like_count",
                    "view_count"
                ],
                ascending=[
                    False,
                    False,
                    False
                ],
                na_position="last"
            )

    # -----------------------------------------------------
    # Convert to dictionaries
    # -----------------------------------------------------

    courses = courses.to_dict(
        "records"
    )

    # -----------------------------------------------------
    # Pagination
    # -----------------------------------------------------

    paginator = Paginator(
        courses,
        20
    )

    page_number = request.GET.get(
        "page"
    )

    page_obj = paginator.get_page(
        page_number
    )

    return render(
        request,
        "home.html",
        {
            "page_obj": page_obj,
            "search": search,
            "sort": sort,
        }
    )


# =========================================================
# STUDENT PROFILE
# =========================================================

@login_required
def student_profile(request):

    profile, created = (
        StudentProfile.objects.get_or_create(
            user=request.user
        )
    )

    if request.method == "POST":

        form = StudentProfileForm(
            request.POST
        )

        if form.is_valid():

            profile.programme = (
                form.cleaned_data[
                    "programme"
                ]
            )

            year = (
                form.cleaned_data[
                    "year_of_study"
                ]
            )

            if year:

                profile.year_of_study = int(
                    year
                )

            else:

                profile.year_of_study = None

            profile.completed_courses = [
                int(course_id)
                for course_id in (
                    form.cleaned_data[
                        "completed_courses"
                    ]
                )
            ]

            profile.save()

            messages.success(
                request,
                "Your student profile has been updated successfully."
            )

            return redirect(
                "student_profile"
            )

    else:

        form = StudentProfileForm(
            initial={
                "programme":
                    profile.programme,

                "year_of_study":
                    (
                        str(
                            profile.year_of_study
                        )
                        if profile.year_of_study
                        else ""
                    ),

                "completed_courses":
                    [
                        str(course_id)
                        for course_id
                        in (
                            profile.completed_courses
                            or []
                        )
                    ]
            }
        )

    return render(
        request,
        "student_profile.html",
        {
            "form": form,
            "profile": profile
        }
    )


# =========================================================
# COURSE DETAIL
# =========================================================

@login_required
def course_detail(
    request,
    course_id
):

    # -----------------------------------------------------
    # Make sure course exists
    # -----------------------------------------------------

    course_matches = data[
        data["course_id"] == course_id
    ]

    if course_matches.empty:

        messages.error(
            request,
            "Course not found."
        )

        return redirect("home")

    # -----------------------------------------------------
    # Record view
    # -----------------------------------------------------

    interaction, created = (
        UserInteraction.objects.get_or_create(
            user=request.user,
            course_id=course_id
        )
    )

    interaction.viewed = True

    interaction.save()

    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    recommendations = (
        get_recommendations(
            course_id,
            data,
            similarity_matrix,
            top_n=5,
            user=request.user
        )
    )

    # -----------------------------------------------------
    # Liked courses
    # -----------------------------------------------------

    liked_course_ids = set(
        UserInteraction.objects.filter(
            user=request.user,
            liked=True
        ).values_list(
            "course_id",
            flat=True
        )
    )

    for item in recommendations:

        item["liked"] = (
            item["course_id"]
            in liked_course_ids
        )

    # -----------------------------------------------------
    # Selected course
    # -----------------------------------------------------

    course = (
        course_matches
        .iloc[0]
        .to_dict()
    )

    current_course_liked = (
        course_id
        in liked_course_ids
    )

    # -----------------------------------------------------
    # Render
    # -----------------------------------------------------

    return render(
        request,
        "course_detail.html",
        {
            "course": course,
            "recommendations": recommendations,
            "current_course_liked":
                current_course_liked
        }
    )


# =========================================================
# LIKE COURSE
# =========================================================

@login_required
def like_course(
    request,
    course_id
):

    if request.method != "POST":

        return redirect("home")

    interaction, created = (
        UserInteraction.objects.get_or_create(
            user=request.user,
            course_id=course_id
        )
    )

    interaction.liked = (
        not interaction.liked
    )

    interaction.save()

    current_course_id = (
        request.POST.get(
            "current_course_id"
        )
    )

    if current_course_id:

        try:

            current_course_id = int(
                current_course_id
            )

            return redirect(
                "course_detail",
                course_id=current_course_id
            )

        except (
            ValueError,
            TypeError
        ):

            pass

    return redirect("home")


# =========================================================
# LIKED COURSES
# =========================================================

@login_required
def liked_courses(request):

    liked_course_ids = set(
        UserInteraction.objects.filter(
            user=request.user,
            liked=True
        ).values_list(
            "course_id",
            flat=True
        )
    )

    liked_courses_data = data[
        data["course_id"].isin(
            liked_course_ids
        )
    ]

    liked_courses = []

    for _, course in (
        liked_courses_data.iterrows()
    ):

        liked_courses.append({

            "course_id":
                int(course["course_id"]),

            "course_name":
                course["course_name"],

            "university":
                course["university"],

            "difficulty":
                course["difficulty"],

            "rating":
                course["rating"],

            "url":
                course["url"]

        })

    return render(
        request,
        "liked_courses.html",
        {
            "liked_courses":
                liked_courses
        }
    )


# =========================================================
# RECORD COURSE VIEW
# =========================================================

def record_course_view(
    user,
    course_id
):

    UserInteraction.objects.update_or_create(
        user=user,
        course_id=course_id,
        defaults={
            "viewed": True
        }
    )
