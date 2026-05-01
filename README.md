# FIFA 2026 Predictor & Player Performance Dashboard

Multi-page Streamlit app with:
- FIFA 2026 match outcome prediction (Win/Draw/Loss) using historical international results.
- Player performance analytics and KMeans clustering using Transfermarkt-style player score data.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data placement

Place CSV files under `data/`:

### Match predictor
- `results.csv`
- `goalscorers.csv`
- `shootouts.csv`

### Player dashboard
- `players.csv`
- `appearances.csv`
- `games.csv`
- `clubs.csv`
- `competitions.csv`

If files are missing, the app shows friendly guidance with Kaggle links.

## Highlights

- Dark themed UI and custom CSS.
- Cached data and cached model training (`@st.cache_data`, `@st.cache_resource`).
- Match models: Random Forest, Gradient Boosting, Logistic Regression.
- Player clustering: StandardScaler + KMeans + PCA + silhouette diagnostics.
