import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("netflix_cleaned.csv", parse_dates=["date_added"])
print(df.shape)
print(df["type"].value_counts())

# 1. Movies vs TV Shows
df["type"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Movies vs TV Shows"); plt.ylabel("")
plt.show()

# 2. Titles added per year
sns.countplot(data=df, x="year_added", hue="type")
plt.xticks(rotation=45); plt.title("Titles added per year")
plt.show()

# 3. Top 10 countries
known = df[df["main_country"] != "Unknown"]
known["main_country"].value_counts().head(10).plot(kind="barh")
plt.title("Top 10 countries"); plt.show()

# 4. Top 10 genres
df["main_genre"].value_counts().head(10).plot(kind="barh")
plt.title("Top 10 genres"); plt.show()

# 5. Ratings
sns.countplot(data=df, y="rating", order=df["rating"].value_counts().index)
plt.title("Ratings"); plt.show()

# 6. Movie duration
movies = df[df["duration_unit"] == "min"]
sns.histplot(movies["duration_value"], bins=30)
plt.title("Movie duration (minutes)"); plt.show()

# 7. TV show seasons
shows = df[df["duration_unit"].str.contains("Season")]
shows["duration_value"].value_counts().sort_index().plot(kind="bar")
plt.title("TV show seasons"); plt.show()

# 8. Best months
order = ["January","February","March","April","May","June","July",
         "August","September","October","November","December"]
df["month_added"].value_counts().reindex(order).plot(kind="bar")
plt.title("Titles added per month"); plt.show()

print(df["type"].value_counts(normalize=True) * 100)
print(df["year_added"].value_counts().head(3))
print(df["month_added"].value_counts().head(3))
