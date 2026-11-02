import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Netflix Dataset.csv")  
 # use your file name
print(df.shape)
df.info()
df.head()
# "Not Given" is really a missing value, so convert it to NaN

df = df.replace(["Not Given", "not given", ""], np.nan)
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("Duplicate show_id:", df["show_id"].duplicated().sum())
df = df.drop_duplicates()
df = df.drop_duplicates(subset="show_id")

# clean text columns
text_cols = df.select_dtypes(include=["object", "string"]).columns
for c in text_cols:
    df[c] = df[c].str.strip()

df["type"] = df["type"].str.title().replace({"Tv Show": "TV Show"})

# fill nulls
df["director"] = df["director"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
df = df.dropna(subset=["date_added", "duration"])

# fix date format
df["date_added"] = pd.to_datetime(df["date_added"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["date_added"])

print(df.isnull().sum())
print(df.shape)
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
df = df.dropna(subset=["date_added", "duration"])

# fix date format (your file uses m/d/yyyy)
df["date_added"] = pd.to_datetime(df["date_added"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["date_added"])
df["year_added"]  = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month_name()

# split "90 min" / "2 Seasons"
df["duration_value"] = df["duration"].str.extract(r"(\d+)").astype(int)
df["duration_unit"]  = df["duration"].str.extract(r"([A-Za-z]+)")
df["main_genre"]   = df["listed_in"].str.split(",").str[0]
df["genre_count"]  = df["listed_in"].str.split(",").str.len()
df["main_country"] = df["country"].str.split(",").str[0]
df["years_to_add"] = df["year_added"] - df["release_year"]   # lag before Netflix added it

# group ratings
audience = {"TV-Y":"Kids","TV-Y7":"Kids","G":"Kids","TV-G":"Kids","PG":"Kids",
            "TV-PG":"Teens","PG-13":"Teens","TV-14":"Teens",
            "R":"Adults","TV-MA":"Adults","NC-17":"Adults"}
df["audience"] = df["rating"].map(audience).fillna("Other")
df["type"] = df["type"].astype("category")
df["type_code"] = df["type"].cat.codes          # Movie=0, TV Show=1

# one-hot encode audience
df = pd.get_dummies(df, columns=["audience"], prefix="aud")

# final checks
print(df.isnull().sum())
print(df.dtypes)
df.to_csv("netflix_cleaned.csv", index=False)


