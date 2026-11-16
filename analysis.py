from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "spotify_dataset.csv"
IMG = ROOT / "images"
RESULTS = ROOT / "results"
IMG.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    print("SPOTIFY MUSIC DATA ANALYSIS")
    print("=" * 45)
    print("Shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates().copy()
    df.to_csv(RESULTS / "spotify_cleaned.csv", index=False)

    # 1. Popularity distribution
    df["track_popularity"].hist(bins=20, edgecolor="black", figsize=(8, 5))
    plt.title("Distribution of Track Popularity")
    plt.xlabel("Popularity"); plt.ylabel("Number of rows")
    plt.tight_layout(); plt.savefig(IMG / "popularity_distribution.png"); plt.close()

    # 2. Top artists
    artists = df["track_artist"].value_counts().head(10)
    artists.sort_values().plot(kind="barh", figsize=(9, 6))
    plt.title("Top 10 Artists by Number of Tracks")
    plt.xlabel("Number of entries"); plt.tight_layout()
    plt.savefig(IMG / "top_10_artists.png"); plt.close()

    # 3. Average popularity by genre
    genre = df.groupby("playlist_genre")["track_popularity"].mean().sort_values(ascending=False)
    genre.plot(kind="bar", figsize=(9, 5))
    plt.title("Average Popularity by Genre")
    plt.ylabel("Average popularity"); plt.xticks(rotation=35, ha="right")
    plt.tight_layout(); plt.savefig(IMG / "genre_popularity.png"); plt.close()

    # 4. Audio-feature correlation
    cols = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "track_popularity"]
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=65, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Audio Feature Correlation")
    plt.tight_layout(); plt.savefig(IMG / "audio_feature_correlation.png"); plt.close()

    # 5. Most popular songs
    top = df.sort_values("track_popularity", ascending=False)[
        ["track_name", "track_artist", "playlist_genre", "track_popularity"]
    ].head(20)
    top.to_csv(RESULTS / "top_20_songs.csv", index=False)
    print("\nTop 10 songs:\n", top.head(10).to_string(index=False))
    print("\nAnalysis complete. Check images/ and results/ folders.")

if __name__ == "__main__":
    main()
