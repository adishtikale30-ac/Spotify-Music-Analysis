from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

def find_dataset():
    files = list(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            "No CSV found. Put your Spotify CSV file inside the dataset folder."
        )
    return files[0]

def find_column(columns, choices):
    lower_map = {str(c).strip().lower(): c for c in columns}
    for choice in choices:
        if choice in lower_map:
            return lower_map[choice]
    for c in columns:
        name = str(c).strip().lower()
        if any(choice in name for choice in choices):
            return c
    return None

def main():
    csv_path = find_dataset()
    df = pd.read_csv(csv_path)
    print("\n===== SPOTIFY MUSIC DATA ANALYSIS =====")
    print(f"\nDataset: {csv_path.name}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    print("\n--- First five rows ---")
    print(df.head().to_string(index=False))
    print("\n--- Column names and data types ---")
    print(df.dtypes.to_string())
    print("\n--- Missing values ---")
    print(df.isnull().sum().to_string())

    duplicate_count = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count}")
    df = df.drop_duplicates().copy()

    # Convert likely numeric fields where possible.
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.8:
                df[col] = converted

    popularity_col = find_column(df.columns, ["track popularity", "popularity"])
    track_col = find_column(df.columns, ["track name", "track", "song name", "song"])
    artist_col = find_column(df.columns, ["artist name", "artists", "artist"])
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if popularity_col:
        df[popularity_col] = pd.to_numeric(df[popularity_col], errors="coerce")
        plt.figure(figsize=(8, 5))
        df[popularity_col].dropna().plot(kind="hist", bins=20, edgecolor="black")
        plt.title("Distribution of Track Popularity")
        plt.xlabel("Popularity")
        plt.ylabel("Number of Tracks")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "popularity_distribution.png", dpi=150)
        plt.close()
        print(f"\nAverage popularity: {df[popularity_col].mean():.2f}")

    if artist_col:
        artist_counts = df[artist_col].astype(str).value_counts().head(10)
        plt.figure(figsize=(10, 6))
        artist_counts.sort_values().plot(kind="barh")
        plt.title("Top 10 Artists by Number of Tracks")
        plt.xlabel("Number of Tracks")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_artists.png", dpi=150)
        plt.close()
        print("\nTop artists by track count:")
        print(artist_counts.to_string())

    if track_col and popularity_col:
        top_tracks = df[[track_col, popularity_col]].dropna()
        top_tracks = top_tracks.sort_values(popularity_col, ascending=False).head(10)
        print("\nTop 10 tracks by popularity:")
        print(top_tracks.to_string(index=False))

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Between Numeric Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_correlation.png", dpi=150)
        plt.close()

    # Save a cleaned copy for inspection.
    df.to_csv(DATA_DIR / "spotify_cleaned.csv", index=False)
    print("\nAnalysis complete.")
    print(f"Charts saved in: {OUTPUT_DIR}")
    print(f"Cleaned data saved as: {DATA_DIR / 'spotify_cleaned.csv'}")

if __name__ == "__main__":
    main()
