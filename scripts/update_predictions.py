import requests, pandas as pd, sqlite3, os
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

OUT_DIR = os.getcwd()  # корень репо

# --- 1) Собираем и обучаем модель для хоккея (как в Colab) ---
# (тот же код сбора, Optuna-оптимизации и расчёта p0_h)
# ... ваш код из Colab сюда скопируйте ...

# --- 2) Собираем upcoming хоккея, отбираем по p0_h и сохраняем ---
# df_up_h.to_csv(os.path.join(OUT_DIR, 'hockey_predictions.csv'), index=False)

# --- 3) Аналогично для тенниса (код из Colab, расчёт p0_t и селекция) ---
# df_up_t.to_csv(os.path.join(OUT_DIR, 'tennis_predictions.csv'), index=False)

print("✅ Predictions updated")
