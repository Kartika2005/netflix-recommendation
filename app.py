import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="Netflix EDA",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 12, "axes.labelsize": 10})

#loading data
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "netflix_titles.csv"
    df = pd.read_csv(csv_path)

    df["director"]    = df["director"].fillna("Unknown") # fill null==unknown
    df["cast"]        = df["cast"].fillna("Unknown")
    df["country"]     = df["country"].fillna("Unknown")
    df["date_added"]  = pd.to_datetime(df["date_added"].str.strip(),
                                        format="%B %d, %Y", errors="coerce")
    df["year_added"]  = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    #extract movie runtime into new column (duration_min)
    df["duration_min"] = (
        df.loc[df["type"] == "Movie", "duration"]
          .str.replace(" min", "", regex=False)
          .astype(float)
    )
    #get tv season counts into seasons
    df["seasons"] = (
        df.loc[df["type"] == "TV Show", "duration"]
          .str.replace(" Season.*", "", regex=True)
          .astype(float)
    )
    
    # Advanced Feature Engineering: Time to Netflix
    df['time_to_netflix'] = df['year_added'] - df['release_year']
    
    # Target Audience Categorization
    def categorize_audience(rating):
        kids = ['TV-Y', 'TV-Y7', 'G', 'TV-G', 'PG', 'TV-Y7-FV']
        teens = ['TV-14', 'PG-13']
        adults = ['TV-MA', 'R', 'NC-17', 'UR', 'NR']
        if rating in kids:
            return 'Kids'
        elif rating in teens:
            return 'Teens'
        elif rating in adults:
            return 'Adults'
        else:
            return 'Unknown'
            
    df['target_audience'] = df['rating'].apply(categorize_audience)
    
    # Smart Missing Data Imputation: Impute Country by Director
    df['country'] = df['country'].replace('Unknown', None) # Temporarily revert
    director_country_map = df.dropna(subset=['director', 'country']).groupby('director')['country'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    df.loc[df['country'].isnull() & df['director'].notnull(), 'country'] = df['director'].map(director_country_map)
    df['country'] = df['country'].fillna('Unknown') # Fill remaining
    
    return df


df = load_data()

# sidebar
st.sidebar.title("Filters")

content_types = ["All"] + sorted(df["type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Content Type", content_types)

min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())
year_range = st.sidebar.slider(
    "Release Year Range",
    min_value=min_year, max_value=max_year,
    value=(2000, max_year)
)

all_ratings = sorted(df["rating"].dropna().unique().tolist())
selected_ratings = st.sidebar.multiselect("Rating(s)", all_ratings, default=all_ratings[:8])

# #filtering data based on sidebar selections
filtered = df.copy()
if selected_type != "All":
    filtered = filtered[filtered["type"] == selected_type]
filtered = filtered[
    (filtered["release_year"] >= year_range[0]) &
    (filtered["release_year"] <= year_range[1])
]
if selected_ratings:
    filtered = filtered[filtered["rating"].isin(selected_ratings)]

# header
st.title("Netflix Movies & TV Shows — EDA Dashboard")
st.caption(
    f"Showing **{len(filtered):,}** titles out of **{len(df):,}** total  |  "
    f"Release years **{year_range[0]}–{year_range[1]}**"
)

# top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles",  f"{len(filtered):,}")
col2.metric("Movies",        f"{(filtered['type']=='Movie').sum():,}")
col3.metric("TV Shows",      f"{(filtered['type']=='TV Show').sum():,}")
col4.metric("Countries",     f"{filtered['country'].str.split(',').explode().str.strip().nunique():,}")

st.divider()

# helper
def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)



r1c1, r1c2 = st.columns(2)

with r1c1:
    st.subheader("Content Type Distribution")
    type_counts = filtered["type"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(type_counts.values, labels=type_counts.index,
           autopct="%1.1f%%", colors=["#E50914", "#221F1F"],
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("Movies vs. TV Shows")
    show_fig(fig)

with r1c2:
    st.subheader("Titles Added Per Year")
    yearly = (
        filtered.dropna(subset=["year_added"])
                .groupby(["year_added", "type"])
                .size()
                .reset_index(name="count")
    )
    yearly = yearly[yearly["year_added"] >= 2010]
    fig, ax = plt.subplots(figsize=(6, 4))
    for ctype, grp in yearly.groupby("type"):
        ax.plot(grp["year_added"], grp["count"], marker="o", label=ctype, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Titles Added")
    ax.set_title("Content Added Over the Years")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    show_fig(fig)

st.divider()


r2c1, r2c2 = st.columns(2)

with r2c1:
    st.subheader("Top 10 Countries")
    country_series = (
        filtered[filtered["country"] != "Unknown"]["country"]
          .str.split(",").explode().str.strip()
    )
    top_countries = country_series.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_countries.index[::-1], top_countries.values[::-1],
            color=sns.color_palette("Set2", 10))
    ax.set_xlabel("Titles")
    ax.set_title("Top 10 Countries by Content")
    for i, v in enumerate(top_countries.values[::-1]):
        ax.text(v + 5, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    show_fig(fig)

with r2c2:
    st.subheader("Content Ratings")
    rating_counts = filtered["rating"].value_counts()
    rating_counts = rating_counts[~rating_counts.index.str.contains("min|Season", na=False)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rating_counts.index, rating_counts.values,
           color=sns.color_palette("tab10", len(rating_counts)))
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings")
    plt.xticks(rotation=45)
    plt.tight_layout()
    show_fig(fig)

st.divider()


r3c1, r3c2 = st.columns(2)

with r3c1:
    st.subheader("Top 10 Genres")
    genre_series = (
        filtered["listed_in"].dropna()
          .str.split(",").explode().str.strip()
    )
    top_genres = genre_series.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_genres.index[::-1], top_genres.values[::-1],
            color=sns.color_palette("husl", 10))
    ax.set_xlabel("Titles")
    ax.set_title("Top 10 Genres")
    for i, v in enumerate(top_genres.values[::-1]):
        ax.text(v + 5, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    show_fig(fig)

with r3c2:
    st.subheader("Top 10 Directors")
    dir_series = (
        filtered[filtered["director"] != "Unknown"]["director"]
          .str.split(",").explode().str.strip()
    )
    top_dirs = dir_series.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top_dirs.index[::-1], top_dirs.values[::-1],
            color=sns.color_palette("muted", 10))
    ax.set_xlabel("Titles")
    ax.set_title("Top 10 Directors")
    for i, v in enumerate(top_dirs.values[::-1]):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    show_fig(fig)

st.divider()

st.subheader("Target Audience & Addition Trends")
r35c1, r35c2 = st.columns(2)

with r35c1:
    audience_counts = filtered["target_audience"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(audience_counts.values, labels=audience_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"), startangle=90)
    ax.set_title("Target Audience Distribution")
    show_fig(fig)

with r35c2:
    if not filtered["time_to_netflix"].dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered["time_to_netflix"].dropna(), bins=20, kde=True, ax=ax, color="#E50914")
        ax.set_title("Time Gap: Release to Netflix Addition")
        ax.set_xlabel("Years")
        ax.set_ylabel("Count")
        show_fig(fig)
    else:
        st.info("No time gap data available.")

st.divider()

st.subheader("Duration Analysis")
r4c1, r4c2 = st.columns(2)

with r4c1:
    movie_dur = filtered.dropna(subset=["duration_min"])
    if not movie_dur.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(movie_dur["duration_min"], bins=30, color="#E50914", edgecolor="white")
        mean_dur = movie_dur["duration_min"].mean()
        ax.axvline(mean_dur, color="black", linestyle="--",
                   label=f"Mean: {mean_dur:.0f} min")
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Count")
        ax.set_title("Movie Duration Distribution")
        ax.legend()
        plt.tight_layout()
        show_fig(fig)
    else:
        st.info("No movie duration data available for the current filters.")

with r4c2:
    tv_seasons = filtered.dropna(subset=["seasons"])
    if not tv_seasons.empty:
        seasons_counts = tv_seasons["seasons"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(seasons_counts.index.astype(int), seasons_counts.values,
               color="#221F1F", width=0.5)
        ax.set_xlabel("Number of Seasons")
        ax.set_ylabel("Count")
        ax.set_title("TV Shows by Season Count")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.tight_layout()
        show_fig(fig)
    else:
        st.info("No TV show data available for the current filters.")

st.divider()


st.subheader("Browse the Dataset")
search_term = st.text_input("Search by title", "")
display_cols = ["title", "type", "director", "country", "release_year", "rating", "listed_in", "duration"]
table_df = filtered[display_cols].copy()
if search_term:
    table_df = table_df[table_df["title"].str.contains(search_term, case=False, na=False)]
st.dataframe(table_df.reset_index(drop=True), use_container_width=True, height=350)

st.caption("Data source: netflix_titles.csv")
