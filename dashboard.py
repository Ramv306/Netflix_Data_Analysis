import pandas as pd
import plotly.express as px

df = pd.read_csv("netflix_cleaned.csv")
figs = []

figs.append(px.pie(df, names="type", title="Movies vs TV Shows"))

yearly = df.groupby(["year_added", "type"]).size().reset_index(name="count")
figs.append(px.bar(yearly, x="year_added", y="count", color="type",
                   barmode="group", title="Titles added per year"))

c = df[df["main_country"] != "Unknown"]["main_country"].value_counts().head(10).reset_index()
c.columns = ["country", "count"]
figs.append(px.bar(c, x="count", y="country", orientation="h", title="Top 10 countries"))

g = df["main_genre"].value_counts().head(10).reset_index()
g.columns = ["genre", "count"]
figs.append(px.bar(g, x="count", y="genre", orientation="h", title="Top 10 genres"))

r = df["rating"].value_counts().reset_index()
r.columns = ["rating", "count"]
figs.append(px.bar(r, x="rating", y="count", title="Ratings"))

with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write("<h1>Netflix Business Insights Dashboard</h1>")
    for i, fig in enumerate(figs):
        f.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))