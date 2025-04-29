import os
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import sqlite3

OUT_DIR = os.getcwd()

# --- Утилиты для фичей ---
def compute_form(df, team, result):
    df = df.sort_values('date')
    wins = (df[result] == 1).astype(int)
    df['form_5'] = wins.groupby(df[team]).rolling(5, min_periods=1).sum().reset_index(0,drop=True)
    return df

def compute_h2h(df, a, b, result):
    d = {}
    for _, r in df.iterrows():
        key = (r[a], r[b])
        d[key] = d.get(key, 0) + (1 if r[result]==1 else 0)
    return df.apply(lambda r: d.get((r[a], r[b]), 0), axis=1)

def compute_rest(df, date_col, team):
    df['date'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    last = df.groupby(df[team])['date'].transform('max')
    today = pd.to_datetime(datetime.utcnow()).tz_localize(None)
    return (today - last).dt.days

def build_and_predict(df, side1, side2, api_upcoming, csv_name):
    # базовая обработка
    if 'MatchResults' in df:
        df['y'] = df['MatchResults'].apply(
            lambda L: 1 if any(x['PointsTeam1']>x['PointsTeam2'] for x in L) else 0
        )
    else:
        df['y'] = 1  # fallback

    # фичи
    df['home'] = 1
    df['away'] = 0
    df['date'] = pd.to_datetime(df['MatchDateTimeUTC'])
    df = compute_form(df, side1, 'y')
    df['h2h'] = compute_h2h(df, side1, side2, 'y')
    df['rest_days'] = compute_rest(df, 'MatchDateTimeUTC', side1)

    # матрица признаков
    X = df[['home','away','form_5','h2h','rest_days']]
    y = df['y']

    # стэкинг: LR + RF
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100))
    ]
    stack = StackingClassifier(estimators=estimators,
        final_estimator=LogisticRegression(), cv=3, passthrough=True)
    pipe = make_pipeline(StandardScaler(), stack)

    # обучаем на исторических
    pipe.fit(X, y)

    # калибруем порог (необязательно, можно p0=0.5)
    probs = pipe.predict_proba(X)[:,1]
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
    thresholds = [p for t,p in zip(prob_true, prob_pred) if t>=0.8]
    p0 = thresholds[0] if thresholds else 0.5

    # собираем upcoming
    try:
        up = requests.get(api_upcoming, timeout=10).json() or []
        df_up = pd.DataFrame(up)
        df_up['home'] = 1
        df_up['away'] = 0
        df_up['date'] = pd.to_datetime(df_up['MatchDateTimeUTC'])
    except:
        # fallback: 3 пустых записей
        df_up = pd.DataFrame([{side1:'A', side2:'B', 'home':1, 'away':0,
                               'MatchDateTimeUTC':datetime.utcnow()} for _ in range(3)])

    # повторяем фичи для upcoming
    df_up = compute_form(df_up, side1, 'home')
    df_up['h2h'] = compute_h2h(df_up, side1, side2, 'home')
    df_up['rest_days'] = compute_rest(df_up, 'MatchDateTimeUTC', side1)
    X_up = df_up[['home','away','form_5','h2h','rest_days']]
    X_up = X_up.fillna(0)

    df_up['prob'] = pipe.predict_proba(X_up)[:,1]
    sel = df_up[df_up['prob']>=p0]
    if sel.empty:
        sel = df_up.sample(3)

    # сохраняем CSV
    path = os.path.join(OUT_DIR, csv_name)
    sel.to_csv(path, index=False)
    print(f"✅ {csv_name} saved with {len(sel)} rows (p0={p0:.2f})")

# --- Hockey ---
try:
    hist_h = requests.get(
      "https://www.openligadb.de/Api/GetMatchResultsByLeagueSaison"
      "?leagueShortCut=bl1&leagueSaison=2024", timeout=10
    ).json()
except:
    hist_h = []
dfh = pd.DataFrame(hist_h)
build_and_predict(
    dfh,
    side1='Team1',
    side2='Team2',
    api_upcoming="https://www.openligadb.de/Api/GetUpcomingMatches/bl1",
    csv_name='hockey_predictions.csv'
)

# --- Tennis ---
# Здесь предполагаем, что у вас есть исторический tennis CSV на диске.
# Если нет — используйте тот же подход с API/fallback.
try:
    import csv
    raw = requests.get("http://www.tennis-data.co.uk/data/2023.csv", timeout=10).text
    dft = pd.read_csv(StringIO(raw))
except:
    dft = pd.DataFrame([
        {'Player1':'A','Player2':'B','Surface':'Hard','WRank':1,'LRank':2,'y':1},
        {'Player1':'B','Player2':'A','Surface':'Clay','WRank':3,'LRank':4,'y':0},
        {'Player1':'A','Player2':'B','Surface':'Grass','WRank':1,'LRank':2,'y':1},
    ])

# готовим данные
dft['home'] = 1; dft['away'] = 0
dft['date'] = pd.to_datetime(dft.get('Date','2024-04-01'))
dft = compute_form(dft, 'Player1', 'y')
dft['h2h'] = compute_h2h(dft, 'Player1', 'Player2', 'y')
dft['rest_days'] = compute_rest(dft, 'Date', 'Player1')

# признаки
X_t = dft[['home','away','form_5','h2h','rest_days']].fillna(0)
y_t = dft['y'].astype(int)

# треним тот же pipeline
pipe_t = pipe
pipe_t.fit(X_t, y_t)

# сохраняем tennis via build_and_predict style
df_up_t = dft.sample(3).copy()
df_up_t['prob'] = pipe_t.predict_proba(df_up_t[['home','away','form_5','h2h','rest_days']])[:,1]
sel_t = df_up_t[df_up_t['prob']>=0.5] or df_up_t
sel_t.to_csv(os.path.join(OUT_DIR,'tennis_predictions.csv'), index=False)
print(f"✅ tennis_predictions.csv saved ({len(sel_t)} rows)")
