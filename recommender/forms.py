from django import forms
from django.contrib.auth.models import User

from .recommendation import data


class RegisterForm(forms.ModelForm):

    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password"
            }
        )
    )

    class Meta:

        model = User

        fields = [
            "username",
            "email",
            "password"
        ]

        widgets = {
            "username": forms.TextInput(
                attrs={
                    "placeholder": "Username"
                }
            ),

            "email": forms.EmailInput(
                attrs={
                    "placeholder": "Email"
                }
            )
        }


class StudentProfileForm(forms.Form):

    programme = forms.CharField(
        max_length=150,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder":
                    "e.g. Bachelor of Data Science"
            }
        )
    )

    year_of_study = forms.ChoiceField(
        choices=[
            (
                "",
                "Select year of study"
            ),
            (
                "1",
                "Year 1"
            ),
            (
                "2",
                "Year 2"
            ),
            (
                "3",
                "Year 3"
            ),
            (
                "4",
                "Year 4"
            ),
        ],
        required=False
    )

    completed_courses = (
        forms.MultipleChoiceField(
            choices=[
                (
                    str(
                        int(row["course_id"])
                    ),
                    (
                        f"{row['course_name']}"
                        f" — "
                        f"{row['university']}"
                    )
                )

                for _, row
                in data.iterrows()
            ],

            widget=forms.CheckboxSelectMultiple,

            required=False
        )
    )
