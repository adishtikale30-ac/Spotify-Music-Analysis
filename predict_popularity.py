from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "spotify_dataset.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
]
TARGET = "track_popularity"

def main():
    df = pd.read_csv(DATA)
    data = df[FEATURES + [TARGET]].dropna()
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("SPOTIFY POPULARITY PREDICTION")
    print(f"Rows used: {len(data)}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared: {r2:.3f}")

    pd.DataFrame([{
        "model": "Random Forest Regressor",
        "target": TARGET,
        "rows_used": len(data),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }]).to_csv(RESULTS / "model_metrics.csv", index=False)

    joblib.dump(model, RESULTS / "spotify_popularity_model.joblib")
    print("Saved model to results/spotify_popularity_model.joblib")

if __name__ == "__main__":
    main()
