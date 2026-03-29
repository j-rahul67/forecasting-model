"""
V3 Pipeline: Per-Restaurant Ensemble (LightGBM + XGBoost + CatBoost) with
Advanced Features + Optuna Tuning. Target: <5% wMAPE on Q4 2025.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import LabelEncoder
import optuna
import warnings, time, sys

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

DATA_PATH  = 'qsr_demand_dataset.csv'
SUBMISSION = 'submission_lgbm.csv'
TEST_START = pd.Timestamp('2025-10-01')
VAL_START  = pd.Timestamp('2024-10-01')
VAL_END    = pd.Timestamp('2024-12-31')

print(f"LightGBM {lgb.__version__} | XGBoost {xgb.__version__} | Optuna {optuna.__version__}")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
t_start = time.time()
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df):,} rows in {time.time()-t_start:.1f}s")

df['quantity_filled'] = df['quantity'].copy()
df.loc[df['date'] >= TEST_START, 'quantity_filled'] = np.nan

train_mask = df['date'] < TEST_START
impute_means = (
    df[train_mask & df['quantity_filled'].notna()]
    .groupby(['restaurant_id', 'menu_item_id', 'day_of_week_num'])['quantity_filled']
    .mean().reset_index().rename(columns={'quantity_filled': '_fill'})
)
df = df.merge(impute_means, on=['restaurant_id', 'menu_item_id', 'day_of_week_num'], how='left')
nan_train = train_mask & df['quantity_filled'].isna()
df.loc[nan_train, 'quantity_filled'] = df.loc[nan_train, '_fill']
df.drop(columns=['_fill'], inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (all 5 domain layers + advanced features)
# ═══════════════════════════════════════════════════════════════════════════════
df = df.sort_values(['restaurant_id', 'menu_item_id', 'date']).reset_index(drop=True)
train_only = df[df['date'] < TEST_START].dropna(subset=['quantity_filled'])

# ── Layer 1: Structural Base ─────────────────────────────────────────────────
for grp_cols, col_name in [
    (['restaurant_id','menu_item_id','month','day_of_week_num'], 'struct_rimdow'),
    (['restaurant_id','menu_item_id','month'], 'struct_rim'),
    (['category','month','day_of_week_num'], 'struct_cmdow'),
]:
    agg = train_only.groupby(grp_cols)['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': col_name})
    df = df.merge(agg, on=grp_cols, how='left')

q4_train = train_only[train_only['month'].isin([10,11,12])]
q4_ri = q4_train.groupby(['restaurant_id','menu_item_id'])['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'q4_hist_ri'})
df = df.merge(q4_ri, on=['restaurant_id','menu_item_id'], how='left')
q4_rimdow = q4_train.groupby(['restaurant_id','menu_item_id','month','day_of_week_num'])['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'struct_rimdow_q4'})
df = df.merge(q4_rimdow, on=['restaurant_id','menu_item_id','month','day_of_week_num'], how='left')
df['struct_rimdow_q4'] = df['struct_rimdow_q4'].fillna(df['struct_rimdow'])
df['q4_hist_ri'] = df['q4_hist_ri'].fillna(df['struct_rim'])

# ── Layer 2: Holiday Calendar ────────────────────────────────────────────────
non_hol_mean = train_only[train_only['is_holiday']==0]['quantity_filled'].mean()
hol_mults = (train_only[train_only['is_holiday']==1].groupby('holiday_name')['quantity_filled'].mean() / non_hol_mean)
df['holiday_mult'] = df['holiday_name'].map(hol_mults.to_dict()).fillna(1.0)

cat_non_hol = train_only[train_only['is_holiday']==0].groupby('category')['quantity_filled'].mean()
cat_hol_mults = {}
for (cat, hol), grp in train_only[train_only['is_holiday']==1].groupby(['category','holiday_name']):
    base = cat_non_hol.get(cat, non_hol_mean)
    if base > 0: cat_hol_mults[(cat, hol)] = grp['quantity_filled'].mean() / base
cat_hol_df = pd.DataFrame([{'category': k[0], 'holiday_name': k[1], 'cat_holiday_mult': v} for k, v in cat_hol_mults.items()])
if len(cat_hol_df) > 0:
    df = df.merge(cat_hol_df, on=['category', 'holiday_name'], how='left')
    df['cat_holiday_mult'] = df['cat_holiday_mult'].fillna(1.0)
else:
    df['cat_holiday_mult'] = 1.0

# ── Layer 3: Weather Elasticity ──────────────────────────────────────────────
cat_elasticity = {}
for cat in train_only['category'].unique():
    c = train_only[train_only['category'] == cat]
    if len(c) > 100:
        cat_elasticity[cat] = np.polyfit(c['avg_temp_f'].values, c['quantity_filled'].values, 1)[0] / c['quantity_filled'].mean()
df['cat_temp_elasticity'] = df['category'].map(cat_elasticity).fillna(0)
q4_avg_temp = train_only[train_only['month'].isin([10,11,12])]['avg_temp_f'].mean()
df['temp_deviation'] = df['avg_temp_f'] - q4_avg_temp
df['weather_impact'] = df['cat_temp_elasticity'] * df['temp_deviation']
le_cat = LabelEncoder()
df['category_enc'] = le_cat.fit_transform(df['category'])
df['cat_temp_interaction'] = df['category_enc'] * df['avg_temp_f']
df['temp_freezing'] = (df['avg_temp_f'] < 32).astype(int)
df['temp_cold'] = ((df['avg_temp_f'] >= 32) & (df['avg_temp_f'] < 50)).astype(int)
df['temp_mild'] = ((df['avg_temp_f'] >= 50) & (df['avg_temp_f'] < 70)).astype(int)
df['has_precip'] = (df['precip_inches'] > 0).astype(int)
df['heavy_precip'] = (df['precip_inches'] > 0.5).astype(int)

# ── Layer 4: Event Lift ──────────────────────────────────────────────────────
non_event_mean = train_only[train_only['is_special_event']==0]['quantity_filled'].mean()
event_mults = (train_only[train_only['is_special_event']==1].groupby('special_event_name')['quantity_filled'].mean() / non_event_mean)
df['event_mult'] = df['special_event_name'].map(event_mults.to_dict()).fillna(1.0)

# ── Layer 5: Store Growth + Promos ───────────────────────────────────────────
store_growth = {}
for r in train_only['restaurant_id'].unique():
    rd = train_only[train_only['restaurant_id'] == r]
    q21, q24 = rd[rd['year']==2021]['quantity_filled'].mean(), rd[rd['year']==2024]['quantity_filled'].mean()
    store_growth[r] = (q24/q21)**(1/3) if q21 > 0 else 1.0
chain_avg = np.mean(list(store_growth.values()))
df['store_growth_mult'] = df['restaurant_id'].map({r: g/chain_avg for r, g in store_growth.items()}).fillna(1.0)

promo_lift = {}
for cat in train_only['category'].unique():
    c = train_only[train_only['category'] == cat]
    on, off = c[c['is_promotion']==1]['quantity_filled'].mean(), c[c['is_promotion']==0]['quantity_filled'].mean()
    if off > 0: promo_lift[cat] = on / off
df['promo_lift_cat'] = np.where(df['is_promotion']==1, df['category'].map(promo_lift).fillna(1.0), 1.0)

# ── Edge Features ────────────────────────────────────────────────────────────
grp = df.groupby(['restaurant_id', 'menu_item_id'])['quantity_filled']
for lag in [7, 14, 364, 365, 728]:
    df[f'lag_{lag}'] = grp.shift(lag)
df['lag_7_filled'] = df['lag_7'].fillna(df['lag_364'])
df['lag_14_filled'] = df['lag_14'].fillna(df['lag_364'])
df['yoy_growth'] = (df['lag_364'] / df['lag_728'].clip(lower=1)).fillna(1.0).clip(0.5, 2.0)

df['rolling_mean_28'] = grp.transform(lambda x: x.shift(1).rolling(28, min_periods=7).mean())
df['rolling_mean_28'] = df['rolling_mean_28'].fillna(df['struct_rimdow'])
df['rolling_std_28'] = grp.transform(lambda x: x.shift(1).rolling(28, min_periods=7).std()).fillna(0)

doy = df['date'].dt.dayofyear
for k in [1, 2, 3]:
    df[f'sin_{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
    df[f'cos_{k}'] = np.cos(2 * np.pi * k * doy / 365.25)

# Q4 Holiday Calendar
def get_thanksgiving(year):
    nov_1 = pd.Timestamp(f'{year}-11-01')
    return nov_1 + pd.Timedelta(days=(3 - nov_1.dayofweek) % 7) + pd.Timedelta(weeks=3)
df['_tg'] = df['year'].map({y: get_thanksgiving(y) for y in range(2021, 2026)})
df['_xm'] = df['year'].map({y: pd.Timestamp(f'{y}-12-25') for y in range(2021, 2026)})
df['days_from_thanksgiving'] = (df['date'] - df['_tg']).dt.days.clip(-60, 60)
df['days_from_christmas'] = (df['date'] - df['_xm']).dt.days.clip(-60, 30)
df['is_thanksgiving_week'] = ((df['days_from_thanksgiving'] >= -3) & (df['days_from_thanksgiving'] <= 3)).astype(int)
df['is_black_friday'] = (df['days_from_thanksgiving'] == 1).astype(int)
df['is_christmas_week'] = ((df['days_from_christmas'] >= -7) & (df['days_from_christmas'] <= 1)).astype(int)
df['is_new_year_period'] = (((df['month'] == 12) & (df['date'].dt.day >= 28)) | ((df['month'] == 1) & (df['date'].dt.day <= 3))).astype(int)
df.drop(columns=['_tg', '_xm'], inplace=True)

# Q4 trend
q4_recent = train_only[(train_only['month'].isin([10,11,12])) & (train_only['year']==2024)]
q4_earlier = train_only[(train_only['month'].isin([10,11,12])) & (train_only['year'] < 2024)]
ri_recent = q4_recent.groupby(['restaurant_id','menu_item_id'])['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled':'q4_recent_mean'})
ri_earlier = q4_earlier.groupby(['restaurant_id','menu_item_id'])['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled':'q4_earlier_mean'})
ri_trend = ri_recent.merge(ri_earlier, on=['restaurant_id','menu_item_id'], how='left')
ri_trend['q4_trend_ratio'] = (ri_trend['q4_recent_mean'] / ri_trend['q4_earlier_mean'].clip(lower=1)).clip(0.5, 2.0)
df = df.merge(ri_trend[['restaurant_id','menu_item_id','q4_trend_ratio']], on=['restaurant_id','menu_item_id'], how='left')
df['q4_trend_ratio'] = df['q4_trend_ratio'].fillna(1.0)
df['struct_rimdow_trended'] = df['struct_rimdow'] * df['q4_trend_ratio']

q4_cmdow = q4_train.groupby(['category','month','day_of_week_num'])['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'q4_cmdow'})
df = df.merge(q4_cmdow, on=['category','month','day_of_week_num'], how='left')
df['q4_cmdow'] = df['q4_cmdow'].fillna(df['struct_cmdow'])

# Interactions
df['struct_x_event'] = df['struct_rimdow'] * df['event_mult']
df['struct_x_holiday'] = df['struct_rimdow'] * df['holiday_mult']
df['yoy_ratio'] = df['lag_364'] / df['struct_rimdow'].clip(lower=1)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
# dow × month interaction
df['dow_x_month'] = df['day_of_week_num'] * 100 + df['month']

# Target encoding: smoothed mean per (restaurant, item, month) with regularization
SMOOTH = 100  # regularization strength
global_mean = train_only['quantity_filled'].mean()
te_rim = train_only.groupby(['restaurant_id','menu_item_id','month']).agg(
    te_mean=('quantity_filled','mean'), te_count=('quantity_filled','count')
).reset_index()
te_rim['target_enc_rim'] = (te_rim['te_count'] * te_rim['te_mean'] + SMOOTH * global_mean) / (te_rim['te_count'] + SMOOTH)
df = df.merge(te_rim[['restaurant_id','menu_item_id','month','target_enc_rim']], on=['restaurant_id','menu_item_id','month'], how='left')
df['target_enc_rim'] = df['target_enc_rim'].fillna(global_mean)

# Target encoding: (restaurant, item, dow)
te_ridow = train_only.groupby(['restaurant_id','menu_item_id','day_of_week_num']).agg(
    te_mean=('quantity_filled','mean'), te_count=('quantity_filled','count')
).reset_index()
te_ridow['target_enc_ridow'] = (te_ridow['te_count'] * te_ridow['te_mean'] + SMOOTH * global_mean) / (te_ridow['te_count'] + SMOOTH)
df = df.merge(te_ridow[['restaurant_id','menu_item_id','day_of_week_num','target_enc_ridow']], on=['restaurant_id','menu_item_id','day_of_week_num'], how='left')
df['target_enc_ridow'] = df['target_enc_ridow'].fillna(global_mean)

# Lagged weather (1-day, 3-day avg)
df['temp_lag1'] = df.groupby(['restaurant_id'])['avg_temp_f'].shift(1)
df['temp_lag1'] = df['temp_lag1'].fillna(df['avg_temp_f'])
df['temp_rolling3'] = df.groupby(['restaurant_id'])['avg_temp_f'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# Payday proximity (1st and 15th)
df['day_of_month'] = df['date'].dt.day
df['days_from_payday'] = df['day_of_month'].apply(lambda d: min(abs(d-1), abs(d-15), abs(d-16)))
df['is_payday_window'] = (df['days_from_payday'] <= 2).astype(int)

# Month position
df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)

# Promo cannibalization: count of concurrent promos in same category on same day
promo_counts = df[df['is_promotion']==1].groupby(['date','category','restaurant_id']).size().reset_index(name='promo_count_cat')
df = df.merge(promo_counts, on=['date','category','restaurant_id'], how='left')
df['promo_count_cat'] = df['promo_count_cat'].fillna(0).astype(int)

# Rolling volatility per item (CV = std/mean)
item_cv = train_only.groupby(['restaurant_id','menu_item_id']).agg(
    cv_mean=('quantity_filled','mean'), cv_std=('quantity_filled','std')
).reset_index()
item_cv['item_cv'] = (item_cv['cv_std'] / item_cv['cv_mean'].clip(lower=1)).clip(0, 2)
df = df.merge(item_cv[['restaurant_id','menu_item_id','item_cv']], on=['restaurant_id','menu_item_id'], how='left')
df['item_cv'] = df['item_cv'].fillna(0.5)

# Encoding
le_rest, le_item, le_prec = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['restaurant_enc'] = le_rest.fit_transform(df['restaurant_id'])
df['menu_item_enc'] = le_item.fit_transform(df['menu_item_id'])
df['precip_type_enc'] = le_prec.fit_transform(df['precip_type'].fillna('None'))
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

print(f"Feature engineering done in {time.time()-t_start:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE LIST
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    # Entity (for global model — removed for per-restaurant)
    'menu_item_enc', 'category_enc',
    # Item attribute
    'unit_price',
    # Calendar
    'month', 'day_of_week_num', 'is_weekend', 'day_of_year', 'week_of_year', 'day_of_month',
    # External signals
    'is_holiday', 'is_special_event', 'is_promotion',
    # Q4 holiday calendar
    'days_from_thanksgiving', 'days_from_christmas',
    'is_thanksgiving_week', 'is_black_friday', 'is_christmas_week', 'is_new_year_period',
    # Weather
    'avg_temp_f', 'precip_inches', 'precip_type_enc',
    'temp_freezing', 'temp_cold', 'temp_mild', 'has_precip', 'heavy_precip',
    # Layer 1: Structural base
    'struct_rimdow', 'struct_rim', 'struct_cmdow', 'q4_hist_ri', 'struct_rimdow_q4',
    # Layer 2: Holiday lift
    'holiday_mult', 'cat_holiday_mult',
    # Layer 3: Weather interactions
    'cat_temp_elasticity', 'temp_deviation', 'weather_impact',
    # Layer 4: Event lift
    'event_mult',
    # Layer 5: Store growth + promos
    'store_growth_mult', 'promo_lift_cat',
    # Stable lags
    'lag_364', 'lag_365', 'lag_7_filled', 'lag_14_filled',
    # Rolling
    'rolling_mean_28', 'rolling_std_28',
    # Fourier
    'sin_1', 'cos_1', 'sin_2', 'cos_2', 'sin_3', 'cos_3',
    # Interactions
    'struct_x_event', 'struct_x_holiday', 'yoy_ratio',
    'cat_temp_interaction',
    # Trend
    'yoy_growth', 'q4_cmdow', 'q4_trend_ratio', 'struct_rimdow_trended',
    # TIER 3: Advanced features
    'dow_x_month',
    'target_enc_rim', 'target_enc_ridow',
    'temp_lag1', 'temp_rolling3',
    'days_from_payday', 'is_payday_window',
    'is_month_start', 'is_month_end',
    'promo_count_cat',
    'item_cv',
]
TARGET = 'quantity_filled'

print(f"Total features: {len(FEATURES)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SPLITS
# ═══════════════════════════════════════════════════════════════════════════════
train1 = df[(df['date'] < VAL_START) & df[TARGET].notna()].copy()
val = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END) & df[TARGET].notna()].copy()
train2 = df[(df['date'] < TEST_START) & df[TARGET].notna()].copy()
test = df[df['date'] >= TEST_START].copy()

def wmape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(y_true[mask])

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1+2+4: PER-RESTAURANT ENSEMBLE WITH OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════════════════════
restaurants = sorted(df['restaurant_id'].unique())
print(f"\n{'='*70}")
print(f"TRAINING PER-RESTAURANT ENSEMBLE (15 restaurants x 3 algorithms)")
print(f"{'='*70}")

all_val_preds = pd.DataFrame()
all_test_preds = pd.DataFrame()

N_OPTUNA_TRIALS = 30  # trials per restaurant for LightGBM tuning

for i, rest in enumerate(restaurants):
    t_rest = time.time()

    # Filter data for this restaurant
    r_train1 = train1[train1['restaurant_id'] == rest]
    r_val = val[val['restaurant_id'] == rest]
    r_train2 = train2[train2['restaurant_id'] == rest]
    r_test = test[test['restaurant_id'] == rest]

    X_tr1, y_tr1 = r_train1[FEATURES], r_train1[TARGET]
    X_val_r, y_val_r = r_val[FEATURES], r_val[TARGET]
    X_tr2, y_tr2 = r_train2[FEATURES], r_train2[TARGET]
    X_test_r = r_test[FEATURES]

    w_tr1 = r_train1['struct_rimdow'].clip(lower=1).values
    w_val_r = r_val['struct_rimdow'].clip(lower=1).values
    w_tr2 = r_train2['struct_rimdow'].clip(lower=1).values

    # ── TIER 4: Optuna tuning for LightGBM ──────────────────────────────────
    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'num_leaves': trial.suggest_int('num_leaves', 63, 511),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': 5,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
            'verbose': -1, 'n_jobs': -1, 'random_state': 42,
        }
        ds_tr = lgb.Dataset(X_tr1, label=y_tr1, weight=w_tr1, free_raw_data=False)
        ds_val = lgb.Dataset(X_val_r, label=y_val_r, weight=w_val_r, free_raw_data=False, reference=ds_tr)
        m = lgb.train(params, ds_tr, num_boost_round=2000,
                      valid_sets=[ds_val], valid_names=['val'],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        preds = np.clip(m.predict(X_val_r), 0, None)
        return wmape(y_val_r, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_lgb_params = study.best_params
    best_lgb_params.update({'objective': 'regression_l1', 'metric': 'mae',
                            'bagging_freq': 5, 'verbose': -1, 'n_jobs': -1, 'random_state': 42})

    # ── Train LightGBM (Phase 1 → best_n, Phase 2 → final) ─────────────────
    ds_tr1 = lgb.Dataset(X_tr1, label=y_tr1, weight=w_tr1, free_raw_data=False)
    ds_val_lgb = lgb.Dataset(X_val_r, label=y_val_r, weight=w_val_r, free_raw_data=False, reference=ds_tr1)
    lgb_m1 = lgb.train(best_lgb_params, ds_tr1, num_boost_round=2000,
                        valid_sets=[ds_val_lgb], valid_names=['val'],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    best_n_lgb = lgb_m1.best_iteration

    ds_tr2 = lgb.Dataset(X_tr2, label=y_tr2, weight=w_tr2, free_raw_data=False)
    lgb_final = lgb.train(best_lgb_params, ds_tr2, num_boost_round=best_n_lgb,
                          callbacks=[lgb.log_evaluation(0)])

    lgb_val_pred = np.clip(lgb_final.predict(X_val_r), 0, None)
    lgb_test_pred = np.clip(lgb_final.predict(X_test_r), 0, None)

    # ── Train XGBoost ────────────────────────────────────────────────────────
    xgb_params = {
        'objective': 'reg:absoluteerror',
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 30,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'n_jobs': -1, 'random_state': 42,
        'verbosity': 0,
    }
    dtr1_xgb = xgb.DMatrix(X_tr1, label=y_tr1, weight=w_tr1)
    dval_xgb = xgb.DMatrix(X_val_r, label=y_val_r, weight=w_val_r)
    xgb_m1 = xgb.train(xgb_params, dtr1_xgb, num_boost_round=2000,
                        evals=[(dval_xgb, 'val')],
                        early_stopping_rounds=30, verbose_eval=False)
    best_n_xgb = xgb_m1.best_iteration

    dtr2_xgb = xgb.DMatrix(X_tr2, label=y_tr2, weight=w_tr2)
    xgb_final = xgb.train(xgb_params, dtr2_xgb, num_boost_round=best_n_xgb, verbose_eval=False)

    xgb_val_pred = np.clip(xgb_final.predict(xgb.DMatrix(X_val_r)), 0, None)
    xgb_test_pred = np.clip(xgb_final.predict(xgb.DMatrix(X_test_r)), 0, None)

    # ── Train CatBoost ───────────────────────────────────────────────────────
    cb_m1 = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=8,
        loss_function='MAE', l2_leaf_reg=3,
        subsample=0.8, colsample_bylevel=0.7,
        random_seed=42, verbose=0,
        early_stopping_rounds=30,
    )
    cb_m1.fit(X_tr1, y_tr1, sample_weight=w_tr1,
              eval_set=(X_val_r, y_val_r), verbose=0)
    best_n_cb = cb_m1.get_best_iteration()

    cb_final = CatBoostRegressor(
        iterations=best_n_cb, learning_rate=0.03, depth=8,
        loss_function='MAE', l2_leaf_reg=3,
        subsample=0.8, colsample_bylevel=0.7,
        random_seed=42, verbose=0,
    )
    cb_final.fit(X_tr2, y_tr2, sample_weight=w_tr2, verbose=0)

    cb_val_pred = np.clip(cb_final.predict(X_val_r), 0, None)
    cb_test_pred = np.clip(cb_final.predict(X_test_r), 0, None)

    # ── Optimize ensemble weights on validation ─────────────────────────────
    best_w, best_val_wmape = (0.33, 0.33, 0.34), 1.0
    for w1 in np.arange(0.1, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05: continue
            ens = w1 * lgb_val_pred + w2 * xgb_val_pred + w3 * cb_val_pred
            w_val = wmape(y_val_r, ens)
            if w_val < best_val_wmape:
                best_val_wmape = w_val
                best_w = (w1, w2, w3)

    # ── Ensemble predictions ─────────────────────────────────────────────────
    ens_val = best_w[0] * lgb_val_pred + best_w[1] * xgb_val_pred + best_w[2] * cb_val_pred
    ens_test = best_w[0] * lgb_test_pred + best_w[1] * xgb_test_pred + best_w[2] * cb_test_pred

    # Bias correction per restaurant
    bias = y_val_r.mean() / ens_val.mean() if ens_val.mean() > 0 else 1.0
    ens_val_corrected = ens_val * bias
    ens_test_corrected = ens_test * bias

    val_wmape_rest = wmape(y_val_r, ens_val_corrected)

    # Store predictions
    val_df = r_val[['date','restaurant_id','menu_item_id','quantity']].copy()
    val_df['predicted'] = ens_val_corrected
    all_val_preds = pd.concat([all_val_preds, val_df])

    test_df = r_test[['date','restaurant_id','menu_item_id']].copy()
    test_df['predicted'] = ens_test_corrected
    if 'quantity' in r_test.columns:
        test_df['quantity'] = r_test['quantity'].values
    all_test_preds = pd.concat([all_test_preds, test_df])

    elapsed = time.time() - t_rest
    print(f"  {rest}: val wMAPE={val_wmape_rest*100:.2f}% | "
          f"w=({best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}) | "
          f"bias={bias:.3f} | n_lgb={best_n_lgb} n_xgb={best_n_xgb} n_cb={best_n_cb} | "
          f"{elapsed:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# OVERALL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")

# Validation (Q4 2024)
val_overall = wmape(all_val_preds['quantity'], all_val_preds['predicted'])
print(f"Q4 2024 Val wMAPE:  {val_overall*100:.2f}%  (accuracy: {(1-val_overall)*100:.2f}%)")

# Test (Q4 2025)
test_with_actual = all_test_preds[all_test_preds['quantity'].notna()].copy()
test_overall = wmape(test_with_actual['quantity'], test_with_actual['predicted'])
print(f"Q4 2025 Test wMAPE: {test_overall*100:.2f}%  (accuracy: {(1-test_overall)*100:.2f}%)")
print(f"Mean actual: {test_with_actual['quantity'].mean():.2f} | predicted: {test_with_actual['predicted'].mean():.2f}")
print(f"{'='*60}")

# Category breakdown
test_wa = test_with_actual.merge(df[['date','restaurant_id','menu_item_id','category','month']].drop_duplicates(),
                                  on=['date','restaurant_id','menu_item_id'], how='left')
mask = test_wa['quantity'] > 0
total_vol = test_wa.loc[mask, 'quantity'].sum()
print(f"\n{'Category':<15s} {'wMAPE':>8s} {'Weight':>8s}")
print("-" * 35)
for cat in sorted(test_wa['category'].unique()):
    m = test_wa['category'] == cat
    cw = wmape(test_wa.loc[m, 'quantity'], test_wa.loc[m, 'predicted'])
    vol = test_wa.loc[m & mask, 'quantity'].sum()
    print(f"{cat:<15s} {cw*100:>7.2f}% {vol/total_vol*100:>7.1f}%")

print()
for mon, name in [(10,'October'), (11,'November'), (12,'December')]:
    m = test_wa['month'] == mon
    print(f"{name:<15s} {wmape(test_wa.loc[m,'quantity'], test_wa.loc[m,'predicted'])*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SUBMISSION CSV
# ═══════════════════════════════════════════════════════════════════════════════
submission = all_test_preds[['date', 'restaurant_id', 'menu_item_id', 'predicted']].copy()
submission = submission.rename(columns={'predicted': 'predicted_quantity'})
submission['date'] = pd.to_datetime(submission['date']).dt.strftime('%Y-%m-%d')
submission = submission.sort_values(['date','restaurant_id','menu_item_id']).reset_index(drop=True)

assert len(submission) == 69000, f"Row count: {len(submission)} (expected 69,000)"
assert submission['predicted_quantity'].min() >= 0, "Negative predictions!"

submission.to_csv(SUBMISSION, index=False)
print(f"\nSubmission saved: {SUBMISSION} ({len(submission):,} rows)")
print(f"Total pipeline time: {time.time()-t_start:.0f}s")
