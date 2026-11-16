# Spotify Music Data Analysis 🎵

## Project overview
This project analyzes the supplied Spotify playlist tracks dataset using Python, Pandas,
Matplotlib and Scikit-learn. It explores popularity, artists, genres and audio features,
then trains a Random Forest regression model to estimate track popularity.

## Dataset details
- Rows: 32,833
- Columns: 23
- Duplicate rows: 0
- Columns include track name, artist, popularity, playlist genre, danceability, energy,
  acousticness, instrumentalness, valence, tempo and duration.

## Project structure
- `dataset/spotify_dataset.csv`: dataset supplied for this project
- `src/analysis.py`: data inspection and visual analysis
- `src/predict_popularity.py`: machine-learning regression model
- `images/`: charts generated from the dataset
- `results/`: summaries, top songs and model metrics

## Run in VS Code
1. Extract the ZIP and open the project folder in VS Code.
2. Open Terminal > New Terminal.
3. Install libraries: `python -m pip install -r requirements.txt`
4. Run analysis: `python src/analysis.py`
5. Run machine learning: `python src/predict_popularity.py`

If `python` does not work on Windows, try `py` instead.

## Machine learning
The target is `track_popularity`. The model uses numeric audio features such as danceability,
energy, loudness, acousticness, valence, tempo and duration. The data is split into training
and testing sets (80% / 20%). Metrics include MAE, RMSE and R-squared.

**Important limitation:** popularity is affected by many factors not present in the audio
features, such as marketing, artist recognition, release timing and playlist exposure.
The model is an educational estimate, not a reliable forecast of future streaming success.
The dataset contains repeated tracks across playlist contexts, so a future improvement is
to split by unique track ID to test generalization to unseen songs.

## Author
Adishti Kale
