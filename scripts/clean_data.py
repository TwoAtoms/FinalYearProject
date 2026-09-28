import pandas as pd
from ftfy import fix_text


# Load raw dataset
data = pd.read_csv("data/Coursera.csv")


# Fix weird unicode characters
data = data.map(
    lambda x: fix_text(x) if isinstance(x, str) else x
)


# Check for replacement characters (�)
bad_character_check = (
    data.astype(str)
    .apply(lambda col: col.str.contains("�", na=False))
    .sum()
)

print("Bad character counts:")
print(bad_character_check)


# Show rows containing bad characters
weird_rows = data[
    data.astype(str)
    .apply(lambda col: col.str.contains("�", na=False))
    .any(axis=1)
]

print("Rows with bad characters:")
print(weird_rows)


# Check specific column if needed
col = "Course Name"

if col in data.columns:
    mask = data[col].str.contains("�", na=False)
    print("Problematic course names:")
    print(data[mask][col].value_counts())


# Check unique values
print("Unique values:")
print(data.nunique())


# Check duplicates
print("Duplicates before removal:")
print(data.duplicated().sum())


# Remove duplicates
data = data.drop_duplicates()


print("Dataset shape after cleaning:")
print(data.shape)


# Final bad character check
bad_cells = (
    data.astype(str)
    .apply(lambda col: col.str.contains("�", na=False))
).sum().sum()

print("Bad cells after cleaning:", bad_cells)


# Save cleaned dataset
data.to_csv(
    "data/cleaned_Coursera.csv",
    index=False
)

print("Cleaning completed!")