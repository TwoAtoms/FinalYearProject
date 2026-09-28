import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import UserInteraction


# =========================================================
# LOAD COURSE DATA
# =========================================================

data = pd.read_csv(
    "data/cleaned_Coursera.csv"
)


# =========================================================
# STANDARDIZE COLUMN NAMES
# =========================================================

data = data.rename(
    columns={
        "Course Name": "course_name",
        "University": "university",
        "Difficulty Level": "difficulty",
        "Course Rating": "rating",
        "Course URL": "url",
        "Course Description": "description",
        "Skills": "skills",
    }
)


# =========================================================
# CREATE COURSE IDs
# =========================================================

data["course_id"] = range(len(data))


# =========================================================
# TEXT CLEANING
# =========================================================

lemmatizer = WordNetLemmatizer()


def clean_for_tags(text):
    """
    Clean text before creating TF-IDF tags.
    """

    text = str(text)

    # Remove replacement characters
    text = re.sub(r'��+', '', text)

    # Remove non-ASCII characters
    text = re.sub(
        r'[^\x00-\x7F]+',
        '',
        text
    )

    # Keep letters and spaces
    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    # Lowercase
    text = text.lower()

    # Lemmatize words
    text = ' '.join(
        [
            lemmatizer.lemmatize(word)
            for word in text.split()
        ]
    )

    return text


# =========================================================
# CLEAN COURSE INFORMATION
# =========================================================

data["clean_name"] = data[
    "course_name"
].fillna("").apply(
    clean_for_tags
)

data["clean_description"] = data[
    "description"
].fillna("").apply(
    clean_for_tags
)

data["clean_skills"] = data[
    "skills"
].fillna("").apply(
    clean_for_tags
)


# =========================================================
# COMBINE COURSE TEXT
# =========================================================

data["tags"] = (
    data["clean_name"]
    + " "
    + data["clean_description"]
    + " "
    + data["clean_skills"]
)


# =========================================================
# TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


tfidf_matrix = vectorizer.fit_transform(
    data["tags"]
)


# =========================================================
# COURSE-TO-COURSE SIMILARITY
# =========================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# =========================================================
# RATING NORMALIZATION
# =========================================================

def normalize_rating(rating):
    """
    Convert a course rating from 0-5
    into a 0-1 score.
    """

    try:

        rating = float(rating)

        # Keep rating inside 0-5
        rating = max(
            0.0,
            min(rating, 5.0)
        )

        return rating / 5.0

    except (
        ValueError,
        TypeError
    ):

        return 0.0


# =========================================================
# GET USER PROFILE SAFELY
# =========================================================

def get_student_profile(user):
    """
    Safely retrieve the student's profile.

    Returns None if the profile does not exist.
    """

    if user is None:
        return None

    try:

        return user.student_profile

    except Exception:

        return None


# =========================================================
# GET COMPLETED COURSE IDS
# =========================================================

def get_completed_course_ids(profile):
    """
    Convert completed course IDs from the profile
    into a set of integers.
    """

    completed_course_ids = set()

    if profile is None:
        return completed_course_ids

    completed_courses = (
        profile.completed_courses
        or []
    )

    for course_id in completed_courses:

        try:

            completed_course_ids.add(
                int(course_id)
            )

        except (
            ValueError,
            TypeError
        ):

            continue

    return completed_course_ids


# =========================================================
# COLLABORATIVE FILTERING
# =========================================================

def build_interaction_matrix():
    """
    Build a User × Course interaction matrix.

    Each interaction is represented as 1.

    An interaction exists when a user:

    - viewed a course
    - liked a course
    - rated a course
    """

    interactions = UserInteraction.objects.all()

    rows = []

    for interaction in interactions:

        has_interaction = (
            bool(interaction.viewed)
            or bool(interaction.liked)
            or interaction.rating is not None
        )

        if has_interaction:

            rows.append({

                "user_id":
                    interaction.user_id,

                "course_id":
                    interaction.course_id,

                "interaction":
                    1

            })


    # -----------------------------------------------------
    # No interaction data
    # -----------------------------------------------------

    if not rows:

        return pd.DataFrame()


    interaction_data = pd.DataFrame(
        rows
    )


    # -----------------------------------------------------
    # User × Course matrix
    # -----------------------------------------------------

    interaction_matrix = (
        interaction_data
        .pivot_table(
            index="user_id",
            columns="course_id",
            values="interaction",
            aggfunc="max",
            fill_value=0
        )
    )


    return interaction_matrix


def get_collaborative_scores(
    user,
    interaction_matrix,
    minimum_users=2
):
    """
    Calculate collaborative-filtering scores.

    Collaborative filtering is only used when there
    is enough interaction data.

    If there is insufficient data, all collaborative
    scores remain 0.

    This prevents the recommendation system from
    depending on collaborative filtering when there
    are only a few users/interactions.
    """

    number_of_courses = len(data)

    collaborative_scores = [
        0.0
    ] * number_of_courses


    # -----------------------------------------------------
    # No interaction matrix
    # -----------------------------------------------------

    if interaction_matrix.empty:

        return collaborative_scores


    # -----------------------------------------------------
    # Not enough users
    # -----------------------------------------------------

    if len(interaction_matrix.index) < minimum_users:

        return collaborative_scores


    # -----------------------------------------------------
    # Current user has no interaction history
    # -----------------------------------------------------

    if user.id not in interaction_matrix.index:

        return collaborative_scores


    # -----------------------------------------------------
    # Calculate user-to-user similarity
    # -----------------------------------------------------

    user_similarity = cosine_similarity(
        interaction_matrix
    )


    user_index = (
        interaction_matrix.index.get_loc(
            user.id
        )
    )


    similarities = user_similarity[
        user_index
    ]


    # -----------------------------------------------------
    # Current user's interaction vector
    # -----------------------------------------------------

    current_user_interactions = (
        interaction_matrix.loc[user.id]
    )


    # -----------------------------------------------------
    # Store scores
    # -----------------------------------------------------

    score_totals = {}

    similarity_totals = {}


    # -----------------------------------------------------
    # Compare current user with other users
    # -----------------------------------------------------

    for other_index, similarity in enumerate(
        similarities
    ):

        # Do not compare user with themselves
        if other_index == user_index:

            continue


        # Ignore users with no similarity
        if similarity <= 0:

            continue


        other_user_id = (
            interaction_matrix.index[
                other_index
            ]
        )


        other_user_interactions = (
            interaction_matrix.loc[
                other_user_id
            ]
        )


        # -------------------------------------------------
        # Look at courses interacted with by similar user
        # -------------------------------------------------

        for course_id, interacted in (
            other_user_interactions.items()
        ):

            if interacted == 0:

                continue


            # -------------------------------------------------
            # Do not recommend courses the current user
            # has already interacted with
            # -------------------------------------------------

            if course_id in (
                current_user_interactions.index
            ):

                if (
                    current_user_interactions[
                        course_id
                    ] > 0
                ):

                    continue


            # -------------------------------------------------
            # Add similarity contribution
            # -------------------------------------------------

            score_totals[course_id] = (
                score_totals.get(
                    course_id,
                    0.0
                )
                + float(similarity)
            )


            similarity_totals[course_id] = (
                similarity_totals.get(
                    course_id,
                    0.0
                )
                + float(abs(similarity))
            )


    # -----------------------------------------------------
    # Normalize collaborative scores
    # -----------------------------------------------------

    for course_id in score_totals:

        denominator = (
            similarity_totals.get(
                course_id,
                0.0
            )
        )

        if denominator <= 0:

            continue


        score = (
            score_totals[course_id]
            /
            denominator
        )


        # -------------------------------------------------
        # Find course index
        # -------------------------------------------------

        matching_indices = data[
            data["course_id"] == course_id
        ].index


        if len(matching_indices) == 0:

            continue


        idx = matching_indices[0]


        collaborative_scores[
            idx
        ] = float(
            max(
                0.0,
                min(score, 1.0)
            )
        )


    return collaborative_scores


# =========================================================
# COURSE-BASED RECOMMENDATIONS
# =========================================================

def get_recommendations(
    course_id,
    data,
    similarity_matrix,
    top_n=5,
    rating_weight=0.10,
    content_weight=0.50,
    completed_weight=0.20,
    programme_weight=0.10,
    difficulty_weight=0.10,
    user=None
):
    """
    Recommend courses similar to the currently viewed
    course.

    Profile information can additionally influence:

    - completed-course similarity
    - programme relevance
    - difficulty suitability
    - rating
    """


    # --------------------------------------------------
    # Find selected course
    # --------------------------------------------------

    course_matches = data[
        data["course_id"] == course_id
    ]


    if course_matches.empty:

        return []


    current_course_index = (
        course_matches.index[0]
    )


    # --------------------------------------------------
    # Get student profile
    # --------------------------------------------------

    profile = get_student_profile(
        user
    )


    # --------------------------------------------------
    # Get completed courses
    # --------------------------------------------------

    completed_course_ids = (
        get_completed_course_ids(
            profile
        )
    )


    # --------------------------------------------------
    # Similarity to current course
    # --------------------------------------------------

    similarity_scores = list(
        enumerate(
            similarity_matrix[
                current_course_index
            ]
        )
    )


    # --------------------------------------------------
    # Completed course indexes
    # --------------------------------------------------

    completed_indices = []

    if completed_course_ids:

        completed_indices = data[
            data["course_id"].isin(
                completed_course_ids
            )
        ].index.tolist()


    # --------------------------------------------------
    # Programme keywords
    # --------------------------------------------------

    programme_keywords = set()


    if (
        profile is not None
        and profile.programme
    ):

        programme_text = clean_for_tags(
            profile.programme
        )


        programme_keywords = set(
            programme_text.split()
        )


        programme_keywords = {
            word
            for word in programme_keywords
            if len(word) > 2
        }


    # --------------------------------------------------
    # Difficulty based on year
    # --------------------------------------------------

    expected_difficulties = set()


    if profile is not None:

        try:

            year = int(
                profile.year_of_study
            )

        except (
            ValueError,
            TypeError
        ):

            year = None


        if year == 1:

            expected_difficulties = {
                "beginner"
            }


        elif year == 2:

            expected_difficulties = {
                "beginner",
                "intermediate"
            }


        elif year == 3:

            expected_difficulties = {
                "intermediate",
                "advanced"
            }


        elif year is not None and year >= 4:

            expected_difficulties = {
                "intermediate",
                "advanced"
            }


    # --------------------------------------------------
    # Build recommendations
    # --------------------------------------------------

    recommendations = []


    for idx, content_similarity in (
        similarity_scores
    ):

        course = data.iloc[idx]


        recommendation_course_id = int(
            course["course_id"]
        )


        # ------------------------------------------------
        # Do not recommend current course
        # ------------------------------------------------

        if (
            recommendation_course_id
            == course_id
        ):

            continue


        # ------------------------------------------------
        # Do not recommend completed courses
        # ------------------------------------------------

        if (
            recommendation_course_id
            in completed_course_ids
        ):

            continue


        # ------------------------------------------------
        # 1. Content similarity
        # ------------------------------------------------

        content_score = float(
            content_similarity
        )


        # ------------------------------------------------
        # 2. Completed-course similarity
        # ------------------------------------------------

        completed_score = 0.0


        if completed_indices:

            completed_scores = (
                similarity_matrix[
                    completed_indices,
                    idx
                ]
            )


            if len(completed_scores) > 0:

                completed_score = float(
                    max(completed_scores)
                )


        # ------------------------------------------------
        # 3. Programme relevance
        # ------------------------------------------------

        programme_score = 0.0


        if programme_keywords:

            course_text = (
                str(course["course_name"])
                + " "
                + str(course["description"])
                + " "
                + str(course["skills"])
            )


            clean_course_text = (
                clean_for_tags(
                    course_text
                )
            )


            course_words = set(
                clean_course_text.split()
            )


            matching_keywords = (
                programme_keywords
                & course_words
            )


            if matching_keywords:

                programme_score = min(
                    len(matching_keywords)
                    /
                    len(programme_keywords),
                    1.0
                )


        # ------------------------------------------------
        # 4. Difficulty suitability
        # ------------------------------------------------

        difficulty_score = 0.0


        course_difficulty = str(
            course["difficulty"]
        ).strip().lower()


        if (
            course_difficulty
            == "not calibrated"
        ):

            difficulty_score = 0.5


        elif expected_difficulties:

            if (
                course_difficulty
                in expected_difficulties
            ):

                difficulty_score = 1.0

            else:

                difficulty_score = 0.0


        # ------------------------------------------------
        # 5. Course rating
        # ------------------------------------------------

        rating_score = normalize_rating(
            course.get(
                "rating",
                0
            )
        )


        # ------------------------------------------------
        # Final score
        # ------------------------------------------------

        final_score = (

            content_score
            * content_weight

            +

            completed_score
            * completed_weight

            +

            programme_score
            * programme_weight

            +

            difficulty_score
            * difficulty_weight

            +

            rating_score
            * rating_weight

        )


        # ------------------------------------------------
        # Explanation
        # ------------------------------------------------

        reasons = []


        if content_score >= 0.50:

            reasons.append(
                "Highly similar to the course you are viewing"
            )

        elif content_score >= 0.30:

            reasons.append(
                "Related to the course you are viewing"
            )


        if completed_score >= 0.50:

            reasons.append(
                "Similar to courses you have completed"
            )


        if programme_score > 0:

            reasons.append(
                "Related to your programme"
            )


        if difficulty_score == 1.0:

            reasons.append(
                "Suitable for your year of study"
            )


        if rating_score >= 0.80:

            reasons.append(
                "Highly rated by learners"
            )


        if not reasons:

            reasons.append(
                "Recommended based on course similarity"
            )


        recommendations.append({

            "course_id":
                recommendation_course_id,

            "course_name":
                course["course_name"],

            "university":
                course["university"],

            "difficulty":
                course["difficulty"],

            "rating":
                course["rating"],

            "similarity":
                content_score,

            "final_score":
                final_score,

            "reasons":
                reasons

        })


    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    recommendations = sorted(
        recommendations,
        key=lambda x: x["final_score"],
        reverse=True
    )


    return recommendations[:top_n]


# =========================================================
# SEARCH RECOMMENDATIONS
# =========================================================

def get_recommendations_from_query(
    query,
    data,
    vectorizer,
    tfidf_matrix,
    top_n=5,
    rating_weight=0.05
):
    """
    Recommend courses based on a search query.
    """

    # --------------------------------------------------
    # Clean query
    # --------------------------------------------------

    clean_query = clean_for_tags(
        query
    )


    # --------------------------------------------------
    # Empty search
    # --------------------------------------------------

    if not clean_query.strip():

        return []


    # --------------------------------------------------
    # Convert query into TF-IDF
    # --------------------------------------------------

    query_vector = vectorizer.transform(
        [clean_query]
    )


    # --------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]


    # --------------------------------------------------
    # Get top courses
    # --------------------------------------------------

    top_indices = (
        similarity_scores
        .argsort()[::-1][:top_n]
    )


    recommendations = []


    for idx in top_indices:

        course = data.iloc[idx]


        rating = normalize_rating(
            course.get(
                "rating",
                0
            )
        )


        final_score = (
            similarity_scores[idx]
            * (1 - rating_weight)
            +
            rating
            * rating_weight
        )


        recommendations.append({

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

            "similarity":
                float(
                    similarity_scores[idx]
                ),

            "final_score":
                float(
                    final_score
                )

        })


    return sorted(
        recommendations,
        key=lambda x: x["final_score"],
        reverse=True
    )


# =========================================================
# PERSONALIZED HYBRID RECOMMENDATIONS
# =========================================================

def get_user_recommendations(
    user,
    data,
    similarity_matrix,
    top_n=5,

    view_weight=0.10,

    like_weight=0.20,

    completed_weight=0.30,

    collaborative_weight=0.25,

    difficulty_weight=0.10,

    rating_weight=0.05
):
    """
    Personalized hybrid recommender.

    Combines:

    1. Courses the user viewed
    2. Courses the user liked
    3. Courses the user completed
    4. Collaborative filtering
    5. Difficulty suitability
    6. Course rating

    If collaborative data is insufficient,
    collaborative_score becomes 0 and the
    other signals continue working.
    """


    # =====================================================
    # 1. GET USER VIEWS
    # =====================================================

    viewed_course_ids = set(
        UserInteraction.objects.filter(
            user=user,
            viewed=True
        ).values_list(
            "course_id",
            flat=True
        )
    )


    # =====================================================
    # 2. GET USER LIKES
    # =====================================================

    liked_course_ids = set(
        UserInteraction.objects.filter(
            user=user,
            liked=True
        ).values_list(
            "course_id",
            flat=True
        )
    )


    # =====================================================
    # 3. GET STUDENT PROFILE
    # =====================================================

    profile = get_student_profile(
        user
    )


    completed_course_ids = (
        get_completed_course_ids(
            profile
        )
    )


    year_of_study = None


    if profile is not None:

        try:

            year_of_study = int(
                profile.year_of_study
            )

        except (
            ValueError,
            TypeError
        ):

            year_of_study = None


    # =====================================================
    # 4. FIND COURSE INDEXES
    # =====================================================

    viewed_indices = data[
        data["course_id"].isin(
            viewed_course_ids
        )
    ].index.tolist()


    liked_indices = data[
        data["course_id"].isin(
            liked_course_ids
        )
    ].index.tolist()


    completed_indices = data[
        data["course_id"].isin(
            completed_course_ids
        )
    ].index.tolist()


    # =====================================================
    # 5. COLLABORATIVE FILTERING
    # =====================================================

    interaction_matrix = (
        build_interaction_matrix()
    )


    collaborative_scores = (
        get_collaborative_scores(
            user,
            interaction_matrix
        )
    )


    # =====================================================
    # 6. INITIALISE SCORE ARRAYS
    # =====================================================

    number_of_courses = len(data)


    view_scores = [
        0.0
    ] * number_of_courses


    like_scores = [
        0.0
    ] * number_of_courses


    completed_scores = [
        0.0
    ] * number_of_courses


    difficulty_scores = [
        0.0
    ] * number_of_courses


    rating_scores = [
        0.0
    ] * number_of_courses


    # =====================================================
    # 7. VIEW SIMILARITY
    # =====================================================

    if viewed_indices:

        calculated_scores = (
            similarity_matrix[
                viewed_indices
            ].mean(axis=0)
        )


        for idx, score in enumerate(
            calculated_scores
        ):

            view_scores[idx] = float(
                score
            )


    # =====================================================
    # 8. LIKE SIMILARITY
    # =====================================================

    if liked_indices:

        calculated_scores = (
            similarity_matrix[
                liked_indices
            ].mean(axis=0)
        )


        for idx, score in enumerate(
            calculated_scores
        ):

            like_scores[idx] = float(
                score
            )


    # =====================================================
    # 9. COMPLETED COURSE SIMILARITY
    # =====================================================

    if completed_indices:

        calculated_scores = (
            similarity_matrix[
                completed_indices
            ].mean(axis=0)
        )


        for idx, score in enumerate(
            calculated_scores
        ):

            completed_scores[idx] = float(
                score
            )


    # =====================================================
    # 10. DIFFICULTY COMPATIBILITY
    # =====================================================

    difficulty_map = {

        1: {
            "beginner"
        },

        2: {
            "beginner",
            "intermediate"
        },

        3: {
            "intermediate",
            "advanced"
        },

        4: {
            "intermediate",
            "advanced"
        }

    }


    suitable_difficulties = (
        difficulty_map.get(
            year_of_study,
            set()
        )
    )


    for idx in range(
        number_of_courses
    ):

        course = data.iloc[idx]


        difficulty = str(
            course["difficulty"]
        ).strip().lower()


        if (
            difficulty
            in suitable_difficulties
        ):

            difficulty_scores[idx] = 1.0


        elif (
            difficulty
            == "not calibrated"
        ):

            # Unknown difficulty should not
            # completely penalize the course.
            difficulty_scores[idx] = 0.5


        else:

            difficulty_scores[idx] = 0.0


    # =====================================================
    # 11. COURSE RATINGS
    # =====================================================

    for idx in range(
        number_of_courses
    ):

        course = data.iloc[idx]


        rating_scores[idx] = (
            normalize_rating(
                course.get(
                    "rating",
                    0
                )
            )
        )


    # =====================================================
    # 12. BUILD RECOMMENDATIONS
    # =====================================================

    recommendations = []


    for idx in range(
        number_of_courses
    ):

        course = data.iloc[idx]


        course_id = int(
            course["course_id"]
        )


        # -------------------------------------------------
        # Do not recommend liked courses
        # -------------------------------------------------

        if (
            course_id
            in liked_course_ids
        ):

            continue


        # -------------------------------------------------
        # Do not recommend completed courses
        # -------------------------------------------------

        if (
            course_id
            in completed_course_ids
        ):

            continue


        # -------------------------------------------------
        # Individual scores
        # -------------------------------------------------

        view_score = (
            view_scores[idx]
        )


        like_score = (
            like_scores[idx]
        )


        completed_score = (
            completed_scores[idx]
        )


        collaborative_score = (
            collaborative_scores[idx]
        )


        difficulty_score = (
            difficulty_scores[idx]
        )


        rating_score = (
            rating_scores[idx]
        )


        # -------------------------------------------------
        # FINAL HYBRID SCORE
        # -------------------------------------------------

        final_score = (

            view_score
            * view_weight

            +

            like_score
            * like_weight

            +

            completed_score
            * completed_weight

            +

            collaborative_score
            * collaborative_weight

            +

            difficulty_score
            * difficulty_weight

            +

            rating_score
            * rating_weight

        )


        # -------------------------------------------------
        # Reduce already-viewed courses
        # -------------------------------------------------

        if (
            course_id
            in viewed_course_ids
        ):

            final_score *= 0.5


        # =================================================
        # EXPLANATION
        # =================================================

        reasons = []


        if completed_score >= 0.30:

            reasons.append(
                "Similar to courses you have completed"
            )


        if like_score >= 0.30:

            reasons.append(
                "Similar to courses you liked"
            )


        if view_score >= 0.30:

            reasons.append(
                "Similar to courses you viewed"
            )


        if collaborative_score >= 0.30:

            reasons.append(
                "Similar to courses interacted with by other users"
            )


        if difficulty_score == 1.0:

            reasons.append(
                "Suitable for your year of study"
            )


        if rating_score >= 0.80:

            reasons.append(
                "Highly rated by learners"
            )


        # -------------------------------------------------
        # Fallback explanation
        # -------------------------------------------------

        if not reasons:

            reasons.append(
                "Recommended based on course content and available profile information"
            )


        # =================================================
        # STORE RECOMMENDATION
        # =================================================

        recommendations.append({

            "course_id":
                course_id,

            "course_name":
                course["course_name"],

            "university":
                course["university"],

            "difficulty":
                course["difficulty"],

            "rating":
                course["rating"],

            "view_score":
                view_score,

            "like_score":
                like_score,

            "completed_score":
                completed_score,

            "collaborative_score":
                collaborative_score,

            "difficulty_score":
                difficulty_score,

            "rating_score":
                rating_score,

            "final_score":
                float(
                    final_score
                ),

            "reasons":
                reasons

        })


    # =====================================================
    # 13. SORT
    # =====================================================

    recommendations = sorted(
        recommendations,
        key=lambda x: x["final_score"],
        reverse=True
    )


    # =====================================================
    # 14. RETURN TOP N
    # =====================================================

    return recommendations[:top_n]
