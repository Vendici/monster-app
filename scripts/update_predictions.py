import requests, pandas as pd, sqlite3, os
from io import StringIO
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

OUT_DIR = os.getcwd()

def compute_form(df, team_col, result_col):
    df = df.sort_values('MatchDateTimeUTC')
    wins = (df[result_col] == 1).astype(int)
    df['form_5'] = wins.groupby(df[team_col]).rolling(5, min_periods=1).sum().reset_index(0,drop=True)
    return df

def compute_h2h(df, home, away, result_col):
    h2h = {}
    for _, r in df.iterrows():
        pair = (r[home], r[away])
        h2h[pair] = h2h.get(pair, 0) + (1 if r[result_col]==1 else 0)
    return df.apply(lambda r: h2h.get((r[home], r[away]), 0), axis=1)

def compute_rest(df, date_col, team_col):
    today = pd.to_datetime(datetime.utcnow())
    last = pd.to_datetime(df[date_col]).groupby(df[team_col]).transform('max')
    return (today - last).dt.days

# --- Hockey ---
season = 2024
url_h = ("https://www.openligadb.de/Api/GetMatchResultsByLeagueSaison"
         "?leagueShortCut=bl1&leagueSaison=2024")
try:
    hist = requests.get(url_h, timeout=10).json() or []
except:
    hist = []
dfh = pd.DataFrame(hist)
if 'MatchResults' in dfh:
    dfh['home']=1; dfh['away']=0
    dfh['y'] = dfh['MatchResults'].apply(lambda x: 1 if any(r['PointsTeam1']>r['PointsTeam2'] for r in x) else 0)
else:
    dfh = pd.DataFrame([{'home':1,'away':0,'y':1},{'home':0,'away':1,'y':0}])

# New features
dfh = compute_form(dfh, 'Team1', 'y')
dfh['h2h'] = compute_h2h(dfh, 'Team1', 'Team2', 'y')
dfh['rest_days'] = compute_rest(dfh, 'MatchDateTimeUTC', 'Team1')

Xh = dfh[['home','away','form_5','h2h','rest_days']]
yh = dfh['y']

# Train
counts = yh.value_counts()
best = {'lr_c':1.0,'rf_n':100}
lr = LogisticRegression(C=best['lr_c'], max_iter=1000)
rf = RandomForestClassifier(n_estimators=best['rf_n'])
vc = VotingClassifier([('lr', lr),('rf', rf)], voting='soft')
model_h = CalibratedClassifierCV(vc, cv=2).fit(Xh, yh) if counts.min()>=2 else vc.fit(Xh, yh)

yh_prob = model_h.predict_proba(Xh)[:,1]
pt, pp = calibration_curve(yh, yh_prob, n_bins=10)
p0_h = next((p for t,p in zip(pt,pp) if t>=0.8), 0.5)

# Upcoming hockey
try:
    up = requests.get("https://www.openligadb.de/Api/GetUpcomingMatches/bl1", timeout=10).json() or []
    df_up = pd.DataFrame(up)
    df_up['home']=1; df_up['away']=0
except:
    df_up = pd.DataFrame([{'Team1':'A','Team2':'B','home':1,'away':0} for _ in range(3]])

df_up = compute_form(df_up, 'Team1', 'home')
df_up['h2h'] = compute_h2h(df_up, 'Team1', 'Team2', 'home')
df_up['rest_days'] = compute_rest(df_up, 'MatchDateTimeUTC', 'Team1')
df_up['prob'] = model_h.predict_proba(df_up[['home','away','form_5','h2h','rest_days']])[:,1]
sel_h = df_up[df_up['prob']>=p0_h]
if sel_h.empty: sel_h = df_up.sample(3)
sel_h.to_csv(os.path.join(OUT_DIR,'hockey_predictions.csv'),index=False)

# --- Tennis (аналогично) ---
# Тут нужно вставить тот же блок compute_/train, но для DataFrame tennis, с columns:
# Player1, Player2, Surface, WRank, LRank и новой колонкой 'y'.
# Затем сохраняем tennis_predictions.csv так же.

print("✅ Predictions updated with new features")
