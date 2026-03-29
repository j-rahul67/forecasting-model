"""
V4 Pipeline: V3 + Cross-Signal Features
  - City/State demand index (foot traffic proxy)
  - Category demand index (shared demand signal)
  - Component group features (combo ↔ individual item relationships)
  - Cross-restaurant signals (same-city/state peer demand)
  - Item family features (size variants, same-base items)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import optuna
import warnings, time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

DATA_PATH  = 'qsr_demand_dataset.csv'
SUBMISSION = 'submission_v4.csv'
TEST_START = pd.Timestamp('2025-10-01')
VAL_START  = pd.Timestamp('2024-10-01')
VAL_END    = pd.Timestamp('2024-12-31')

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

df = df.sort_values(['restaurant_id', 'menu_item_id', 'date']).reset_index(drop=True)
train_only = df[df['date'] < TEST_START].dropna(subset=['quantity_filled'])

# ═══════════════════════════════════════════════════════════════════════════════
# V3 FEATURES (carried over)
# ═══════════════════════════════════════════════════════════════════════════════
print("Building V3 features...")

# Layer 1: Structural Base
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

# Layer 2: Holiday
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

# Layer 3: Weather
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

# Layer 4: Events
non_event_mean = train_only[train_only['is_special_event']==0]['quantity_filled'].mean()
event_mults = (train_only[train_only['is_special_event']==1].groupby('special_event_name')['quantity_filled'].mean() / non_event_mean)
df['event_mult'] = df['special_event_name'].map(event_mults.to_dict()).fillna(1.0)

# Layer 5: Store growth + promos
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

# Edge features
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

# Q4 holiday calendar
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

df['struct_x_event'] = df['struct_rimdow'] * df['event_mult']
df['struct_x_holiday'] = df['struct_rimdow'] * df['holiday_mult']
df['yoy_ratio'] = df['lag_364'] / df['struct_rimdow'].clip(lower=1)

# Tier 3 advanced
df['dow_x_month'] = df['day_of_week_num'] * 100 + df['month']
global_mean = train_only['quantity_filled'].mean()
SMOOTH = 100
te_rim = train_only.groupby(['restaurant_id','menu_item_id','month']).agg(
    te_mean=('quantity_filled','mean'), te_count=('quantity_filled','count')).reset_index()
te_rim['target_enc_rim'] = (te_rim['te_count'] * te_rim['te_mean'] + SMOOTH * global_mean) / (te_rim['te_count'] + SMOOTH)
df = df.merge(te_rim[['restaurant_id','menu_item_id','month','target_enc_rim']], on=['restaurant_id','menu_item_id','month'], how='left')
df['target_enc_rim'] = df['target_enc_rim'].fillna(global_mean)

te_ridow = train_only.groupby(['restaurant_id','menu_item_id','day_of_week_num']).agg(
    te_mean=('quantity_filled','mean'), te_count=('quantity_filled','count')).reset_index()
te_ridow['target_enc_ridow'] = (te_ridow['te_count'] * te_ridow['te_mean'] + SMOOTH * global_mean) / (te_ridow['te_count'] + SMOOTH)
df = df.merge(te_ridow[['restaurant_id','menu_item_id','day_of_week_num','target_enc_ridow']], on=['restaurant_id','menu_item_id','day_of_week_num'], how='left')
df['target_enc_ridow'] = df['target_enc_ridow'].fillna(global_mean)

df['temp_lag1'] = df.groupby(['restaurant_id'])['avg_temp_f'].shift(1).fillna(df['avg_temp_f'])
df['temp_rolling3'] = df.groupby(['restaurant_id'])['avg_temp_f'].transform(lambda x: x.rolling(3, min_periods=1).mean())

df['day_of_month'] = df['date'].dt.day
df['days_from_payday'] = df['day_of_month'].apply(lambda d: min(abs(d-1), abs(d-15), abs(d-16)))
df['is_payday_window'] = (df['days_from_payday'] <= 2).astype(int)
df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)

promo_counts = df[df['is_promotion']==1].groupby(['date','category','restaurant_id']).size().reset_index(name='promo_count_cat')
df = df.merge(promo_counts, on=['date','category','restaurant_id'], how='left')
df['promo_count_cat'] = df['promo_count_cat'].fillna(0).astype(int)

item_cv = train_only.groupby(['restaurant_id','menu_item_id']).agg(
    cv_mean=('quantity_filled','mean'), cv_std=('quantity_filled','std')).reset_index()
item_cv['item_cv'] = (item_cv['cv_std'] / item_cv['cv_mean'].clip(lower=1)).clip(0, 2)
df = df.merge(item_cv[['restaurant_id','menu_item_id','item_cv']], on=['restaurant_id','menu_item_id'], how='left')
df['item_cv'] = df['item_cv'].fillna(0.5)

le_rest, le_item, le_prec = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['restaurant_enc'] = le_rest.fit_transform(df['restaurant_id'])
df['menu_item_enc'] = le_item.fit_transform(df['menu_item_id'])
df['precip_type_enc'] = le_prec.fit_transform(df['precip_type'].fillna('None'))
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

print(f"V3 features done in {time.time()-t_start:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# V4 NEW FEATURES: Cross-Signal Engineering
# ═══════════════════════════════════════════════════════════════════════════════
print("Building V4 cross-signal features...")

# ── Geography mapping ──────────────────────────────────────────────────────────
geo = df.drop_duplicates('restaurant_id')[['restaurant_id', 'city', 'state']].copy()
rest_city = dict(zip(geo['restaurant_id'], geo['city']))
rest_state = dict(zip(geo['restaurant_id'], geo['state']))
df['city'] = df['restaurant_id'].map(rest_city)
df['state'] = df['restaurant_id'].map(rest_state)

le_city = LabelEncoder()
le_state = LabelEncoder()
df['city_enc'] = le_city.fit_transform(df['city'])
df['state_enc'] = le_state.fit_transform(df['state'])

# ── Feature A: City/State demand index (foot traffic proxy) ───────────────────
# Historical avg daily demand per city per dow per month (from training only)
city_demand = train_only.copy()
city_demand['city'] = city_demand['restaurant_id'].map(rest_city)
city_demand['state'] = city_demand['restaurant_id'].map(rest_state)

# City-level demand index: avg total demand per city per month per dow
city_idx = city_demand.groupby(['city', 'month', 'day_of_week_num'])['quantity_filled'].mean().reset_index()
city_idx.rename(columns={'quantity_filled': 'city_demand_idx'}, inplace=True)
df = df.merge(city_idx, on=['city', 'month', 'day_of_week_num'], how='left')
df['city_demand_idx'] = df['city_demand_idx'].fillna(global_mean)

# State-level demand index
state_idx = city_demand.groupby(['state', 'month', 'day_of_week_num'])['quantity_filled'].mean().reset_index()
state_idx.rename(columns={'quantity_filled': 'state_demand_idx'}, inplace=True)
df = df.merge(state_idx, on=['state', 'month', 'day_of_week_num'], how='left')
df['state_demand_idx'] = df['state_demand_idx'].fillna(global_mean)

# Restaurant's share of its city demand (stable structural ratio)
rest_city_share = train_only.copy()
rest_city_share['city'] = rest_city_share['restaurant_id'].map(rest_city)
city_total = rest_city_share.groupby(['city', 'month'])['quantity_filled'].sum().reset_index().rename(columns={'quantity_filled': 'city_total'})
rest_total = rest_city_share.groupby(['restaurant_id', 'month'])['quantity_filled'].sum().reset_index().rename(columns={'quantity_filled': 'rest_total'})
rest_total['city'] = rest_total['restaurant_id'].map(rest_city)
rest_share = rest_total.merge(city_total, on=['city', 'month'])
rest_share['rest_city_share'] = rest_share['rest_total'] / rest_share['city_total'].clip(lower=1)
df = df.merge(rest_share[['restaurant_id', 'month', 'rest_city_share']], on=['restaurant_id', 'month'], how='left')
df['rest_city_share'] = df['rest_city_share'].fillna(0.5)

# Number of restaurants in same city/state (competition intensity)
city_count = geo.groupby('city').size().reset_index(name='n_restaurants_city')
state_count = geo.groupby('state').size().reset_index(name='n_restaurants_state')
df = df.merge(city_count, on='city', how='left')
df = df.merge(state_count, on='state', how='left')

# ── Feature B: Category demand index ──────────────────────────────────────────
# How is this category doing across the whole chain on this type of day?
cat_chain_idx = train_only.groupby(['category', 'month', 'day_of_week_num'])['quantity_filled'].mean().reset_index()
cat_chain_idx.rename(columns={'quantity_filled': 'cat_chain_demand_idx'}, inplace=True)
df = df.merge(cat_chain_idx, on=['category', 'month', 'day_of_week_num'], how='left')
df['cat_chain_demand_idx'] = df['cat_chain_demand_idx'].fillna(global_mean)

# Item's share within its category at this restaurant
item_cat_share = train_only.groupby(['restaurant_id', 'category', 'month'])['quantity_filled'].sum().reset_index().rename(columns={'quantity_filled': 'cat_total_rest'})
item_share = train_only.groupby(['restaurant_id', 'menu_item_id', 'month'])['quantity_filled'].sum().reset_index().rename(columns={'quantity_filled': 'item_total'})
item_share = item_share.merge(df[['menu_item_id', 'category']].drop_duplicates(), on='menu_item_id')
item_share = item_share.merge(item_cat_share, on=['restaurant_id', 'category', 'month'])
item_share['item_category_share'] = item_share['item_total'] / item_share['cat_total_rest'].clip(lower=1)
df = df.merge(item_share[['restaurant_id', 'menu_item_id', 'month', 'item_category_share']], on=['restaurant_id', 'menu_item_id', 'month'], how='left')
df['item_category_share'] = df['item_category_share'].fillna(0.1)

# Category's share of restaurant total
rest_cat_share = train_only.groupby(['restaurant_id', 'month'])['quantity_filled'].sum().reset_index().rename(columns={'quantity_filled': 'rest_monthly_total'})
cat_rest_share = item_cat_share.merge(rest_cat_share, on=['restaurant_id', 'month'])
cat_rest_share['cat_rest_share'] = cat_rest_share['cat_total_rest'] / cat_rest_share['rest_monthly_total'].clip(lower=1)
df = df.merge(cat_rest_share[['restaurant_id', 'category', 'month', 'cat_rest_share']], on=['restaurant_id', 'category', 'month'], how='left')
df['cat_rest_share'] = df['cat_rest_share'].fillna(0.125)

# ── Feature C: Component / Item family groups ─────────────────────────────────
# Define component groups: items sharing base ingredients
component_groups = {
    # Burger patty family
    'burger_patty': ['M01', 'M02', 'M04', 'M05', 'M06', 'M07', 'M08', 'M43', 'M45'],
    # Chicken family
    'chicken': ['M09', 'M10', 'M11', 'M12', 'M13', 'M14', 'M44', 'M46'],
    # Fries/potato family
    'potato': ['M15', 'M16', 'M22', 'M34'],
    # Drink family
    'drinks': ['M23', 'M24', 'M25', 'M26', 'M29', 'M30'],
    # Milkshake/dessert cold family
    'frozen_dessert': ['M27', 'M28', 'M41', 'M42'],
    # Baked dessert
    'baked_dessert': ['M38', 'M39', 'M40'],
    # Breakfast
    'breakfast': ['M31', 'M32', 'M33', 'M34', 'M35', 'M36', 'M37'],
    # Combos (overlap intentional — combos bridge categories)
    'combo': ['M43', 'M44', 'M45', 'M46', 'M47'],
}

# Map each item to its primary component group
item_to_group = {}
for group, items in component_groups.items():
    for item in items:
        if item not in item_to_group:  # first assignment wins (primary group)
            item_to_group[item] = group
# Fallback for any unmapped
for item in df['menu_item_id'].unique():
    if item not in item_to_group:
        item_to_group[item] = 'other'

df['component_group'] = df['menu_item_id'].map(item_to_group)
le_comp = LabelEncoder()
df['component_group_enc'] = le_comp.fit_transform(df['component_group'])

# Component group demand index: avg demand for this group per restaurant per month per dow
comp_demand = train_only.copy()
comp_demand['component_group'] = comp_demand['menu_item_id'].map(item_to_group)
comp_idx = comp_demand.groupby(['restaurant_id', 'component_group', 'month', 'day_of_week_num'])['quantity_filled'].mean().reset_index()
comp_idx.rename(columns={'quantity_filled': 'component_demand_idx'}, inplace=True)
df = df.merge(comp_idx, on=['restaurant_id', 'component_group', 'month', 'day_of_week_num'], how='left')
df['component_demand_idx'] = df['component_demand_idx'].fillna(global_mean)

# Item's share within its component group
item_comp_total = comp_demand.groupby(['restaurant_id', 'component_group', 'month'])['quantity_filled'].sum().reset_index().rename(columns={'quantity_filled': 'comp_total'})
item_in_comp = comp_demand.groupby(['restaurant_id', 'menu_item_id', 'component_group', 'month'])['quantity_filled'].sum().reset_index().rename(columns={'quantity_filled': 'item_comp_qty'})
item_in_comp = item_in_comp.merge(item_comp_total, on=['restaurant_id', 'component_group', 'month'])
item_in_comp['item_component_share'] = item_in_comp['item_comp_qty'] / item_in_comp['comp_total'].clip(lower=1)
df = df.merge(item_in_comp[['restaurant_id', 'menu_item_id', 'month', 'item_component_share']], on=['restaurant_id', 'menu_item_id', 'month'], how='left')
df['item_component_share'] = df['item_component_share'].fillna(0.1)

# ── Feature D: Size variant ratios ────────────────────────────────────────────
# Pairs of (small, large) items — ratio of demand
size_pairs = [
    ('M15', 'M16'),  # Regular Fries vs Large Fries
    ('M23', 'M24'),  # Regular Drink vs Large Drink
    ('M11', 'M12'),  # Nuggets 6pc vs 12pc
]
for small, large in size_pairs:
    sm_avg = train_only[train_only['menu_item_id'] == small].groupby(['restaurant_id', 'month'])['quantity_filled'].mean().reset_index()
    lg_avg = train_only[train_only['menu_item_id'] == large].groupby(['restaurant_id', 'month'])['quantity_filled'].mean().reset_index()
    merged_sz = sm_avg.merge(lg_avg, on=['restaurant_id', 'month'], suffixes=('_sm', '_lg'))
    merged_sz[f'size_ratio_{small}_{large}'] = merged_sz['quantity_filled_sm'] / merged_sz['quantity_filled_lg'].clip(lower=1)
    df = df.merge(merged_sz[['restaurant_id', 'month', f'size_ratio_{small}_{large}']], on=['restaurant_id', 'month'], how='left')
    df[f'size_ratio_{small}_{large}'] = df[f'size_ratio_{small}_{large}'].fillna(1.0)

# ── Feature E: Combo-to-component ratios ──────────────────────────────────────
combo_map = {
    'M43': 'M01',  # Combo #1 → Cheeseburger
    'M44': 'M09',  # Combo #2 → Chicken Sandwich
    'M45': 'M02',  # Combo #3 → Double Burger
    'M46': 'M11',  # Combo #4 → Nuggets
}
for combo_id, comp_id in combo_map.items():
    combo_avg = train_only[train_only['menu_item_id'] == combo_id].groupby(['restaurant_id', 'month'])['quantity_filled'].mean().reset_index()
    comp_avg = train_only[train_only['menu_item_id'] == comp_id].groupby(['restaurant_id', 'month'])['quantity_filled'].mean().reset_index()
    merged_cc = combo_avg.merge(comp_avg, on=['restaurant_id', 'month'], suffixes=('_combo', '_comp'))
    merged_cc[f'combo_ratio_{combo_id}'] = merged_cc['quantity_filled_combo'] / merged_cc['quantity_filled_comp'].clip(lower=1)
    df = df.merge(merged_cc[['restaurant_id', 'month', f'combo_ratio_{combo_id}']], on=['restaurant_id', 'month'], how='left')
    df[f'combo_ratio_{combo_id}'] = df[f'combo_ratio_{combo_id}'].fillna(1.0)

# ── Feature F: Cross-restaurant peer signals (lagged) ─────────────────────────
# For restaurants in same state: what was the avg demand for this item at peer stores last year?
peer_demand = train_only.copy()
peer_demand['state'] = peer_demand['restaurant_id'].map(rest_state)

# State-level item demand (excluding self) — use yearly averages as stable signal
state_item_demand = peer_demand.groupby(['state', 'menu_item_id', 'month', 'day_of_week_num'])['quantity_filled'].mean().reset_index()
state_item_demand.rename(columns={'quantity_filled': 'state_item_demand_idx'}, inplace=True)
df = df.merge(state_item_demand, on=['state', 'menu_item_id', 'month', 'day_of_week_num'], how='left')
df['state_item_demand_idx'] = df['state_item_demand_idx'].fillna(global_mean)

# Ratio: this restaurant's item demand vs state average for same item
rest_item_avg = train_only.groupby(['restaurant_id', 'menu_item_id', 'month'])['quantity_filled'].mean().reset_index()
rest_item_avg.rename(columns={'quantity_filled': 'rest_item_avg'}, inplace=True)
rest_item_avg['state'] = rest_item_avg['restaurant_id'].map(rest_state)
state_item_avg = peer_demand.groupby(['state', 'menu_item_id', 'month'])['quantity_filled'].mean().reset_index()
state_item_avg.rename(columns={'quantity_filled': 'state_item_avg'}, inplace=True)
rest_vs_state = rest_item_avg.merge(state_item_avg, on=['state', 'menu_item_id', 'month'])
rest_vs_state['rest_vs_state_ratio'] = rest_vs_state['rest_item_avg'] / rest_vs_state['state_item_avg'].clip(lower=1)
df = df.merge(rest_vs_state[['restaurant_id', 'menu_item_id', 'month', 'rest_vs_state_ratio']], on=['restaurant_id', 'menu_item_id', 'month'], how='left')
df['rest_vs_state_ratio'] = df['rest_vs_state_ratio'].fillna(1.0)

# ── Feature G: Price tier features ────────────────────────────────────────────
# Price relative to category average
cat_avg_price = df.drop_duplicates('menu_item_id').groupby('category')['unit_price'].mean().to_dict()
df['cat_avg_price'] = df['category'].map(cat_avg_price)
df['price_vs_cat_avg'] = df['unit_price'] / df['cat_avg_price'].clip(lower=1)

# Price tier (budget/mid/premium within category)
df['price_tier'] = pd.qcut(df['unit_price'], q=3, labels=[0, 1, 2]).astype(int)

print(f"V4 cross-signal features done in {time.time()-t_start:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE LIST (V3 + V4 new)
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    # ── V3 Features ───────────────────────────────────────────────────────────
    'menu_item_enc', 'category_enc', 'unit_price',
    'month', 'day_of_week_num', 'is_weekend', 'day_of_year', 'week_of_year', 'day_of_month',
    'is_holiday', 'is_special_event', 'is_promotion',
    'days_from_thanksgiving', 'days_from_christmas',
    'is_thanksgiving_week', 'is_black_friday', 'is_christmas_week', 'is_new_year_period',
    'avg_temp_f', 'precip_inches', 'precip_type_enc',
    'temp_freezing', 'temp_cold', 'temp_mild', 'has_precip', 'heavy_precip',
    'struct_rimdow', 'struct_rim', 'struct_cmdow', 'q4_hist_ri', 'struct_rimdow_q4',
    'holiday_mult', 'cat_holiday_mult',
    'cat_temp_elasticity', 'temp_deviation', 'weather_impact',
    'event_mult', 'store_growth_mult', 'promo_lift_cat',
    'lag_364', 'lag_365', 'lag_7_filled', 'lag_14_filled',
    'rolling_mean_28', 'rolling_std_28',
    'sin_1', 'cos_1', 'sin_2', 'cos_2', 'sin_3', 'cos_3',
    'struct_x_event', 'struct_x_holiday', 'yoy_ratio', 'cat_temp_interaction',
    'yoy_growth', 'q4_cmdow', 'q4_trend_ratio', 'struct_rimdow_trended',
    'dow_x_month', 'target_enc_rim', 'target_enc_ridow',
    'temp_lag1', 'temp_rolling3',
    'days_from_payday', 'is_payday_window', 'is_month_start', 'is_month_end',
    'promo_count_cat', 'item_cv',
    # ── V4 New: Cross-Signal Features ─────────────────────────────────────────
    # A: City/State foot traffic proxy
    'city_enc', 'state_enc',
    'city_demand_idx', 'state_demand_idx',
    'rest_city_share',
    'n_restaurants_city', 'n_restaurants_state',
    # B: Category demand index
    'cat_chain_demand_idx', 'item_category_share', 'cat_rest_share',
    # C: Component group features
    'component_group_enc', 'component_demand_idx', 'item_component_share',
    # D: Size variant ratios
    'size_ratio_M15_M16', 'size_ratio_M23_M24', 'size_ratio_M11_M12',
    # E: Combo-to-component ratios
    'combo_ratio_M43', 'combo_ratio_M44', 'combo_ratio_M45', 'combo_ratio_M46',
    # F: Cross-restaurant peer signals
    'state_item_demand_idx', 'rest_vs_state_ratio',
    # G: Price tier
    'price_vs_cat_avg', 'price_tier',
]

TARGET = 'quantity_filled'
print(f"Total features: {len(FEATURES)} (V3: 70, V4 new: {len(FEATURES)-70})")

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
# PER-RESTAURANT ENSEMBLE WITH OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════
restaurants = sorted(df['restaurant_id'].unique())
print(f"\n{'='*70}")
print(f"V4: TRAINING PER-RESTAURANT ENSEMBLE (15 × 3 algorithms)")
print(f"{'='*70}")

all_val_preds = pd.DataFrame()
all_test_preds = pd.DataFrame()
N_OPTUNA_TRIALS = 30

for i, rest in enumerate(restaurants):
    t_rest = time.time()

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

    # Optuna for LightGBM
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae',
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

    # Train LightGBM
    ds_tr1 = lgb.Dataset(X_tr1, label=y_tr1, weight=w_tr1, free_raw_data=False)
    ds_val_lgb = lgb.Dataset(X_val_r, label=y_val_r, weight=w_val_r, free_raw_data=False, reference=ds_tr1)
    lgb_m1 = lgb.train(best_lgb_params, ds_tr1, num_boost_round=2000,
                        valid_sets=[ds_val_lgb], valid_names=['val'],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    best_n_lgb = lgb_m1.best_iteration

    ds_tr2 = lgb.Dataset(X_tr2, label=y_tr2, weight=w_tr2, free_raw_data=False)
    lgb_final = lgb.train(best_lgb_params, ds_tr2, num_boost_round=best_n_lgb, callbacks=[lgb.log_evaluation(0)])
    lgb_val_pred = np.clip(lgb_final.predict(X_val_r), 0, None)
    lgb_test_pred = np.clip(lgb_final.predict(X_test_r), 0, None)

    # Train XGBoost
    xgb_params = {
        'objective': 'reg:absoluteerror', 'max_depth': 8, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 30,
        'reg_alpha': 0.2, 'reg_lambda': 0.2, 'n_jobs': -1, 'random_state': 42, 'verbosity': 0,
    }
    dtr1_xgb = xgb.DMatrix(X_tr1, label=y_tr1, weight=w_tr1)
    dval_xgb = xgb.DMatrix(X_val_r, label=y_val_r, weight=w_val_r)
    xgb_m1 = xgb.train(xgb_params, dtr1_xgb, num_boost_round=2000,
                        evals=[(dval_xgb, 'val')], early_stopping_rounds=30, verbose_eval=False)
    best_n_xgb = xgb_m1.best_iteration

    dtr2_xgb = xgb.DMatrix(X_tr2, label=y_tr2, weight=w_tr2)
    xgb_final = xgb.train(xgb_params, dtr2_xgb, num_boost_round=best_n_xgb, verbose_eval=False)
    xgb_val_pred = np.clip(xgb_final.predict(xgb.DMatrix(X_val_r)), 0, None)
    xgb_test_pred = np.clip(xgb_final.predict(xgb.DMatrix(X_test_r)), 0, None)

    # Train CatBoost
    cb_m1 = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=8,
        loss_function='MAE', l2_leaf_reg=3, subsample=0.8, colsample_bylevel=0.7,
        random_seed=42, verbose=0, early_stopping_rounds=30,
    )
    cb_m1.fit(X_tr1, y_tr1, sample_weight=w_tr1, eval_set=(X_val_r, y_val_r), verbose=0)
    best_n_cb = cb_m1.get_best_iteration()

    cb_final = CatBoostRegressor(
        iterations=best_n_cb, learning_rate=0.03, depth=8,
        loss_function='MAE', l2_leaf_reg=3, subsample=0.8, colsample_bylevel=0.7,
        random_seed=42, verbose=0,
    )
    cb_final.fit(X_tr2, y_tr2, sample_weight=w_tr2, verbose=0)
    cb_val_pred = np.clip(cb_final.predict(X_val_r), 0, None)
    cb_test_pred = np.clip(cb_final.predict(X_test_r), 0, None)

    # Optimize ensemble weights
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

    ens_val = best_w[0] * lgb_val_pred + best_w[1] * xgb_val_pred + best_w[2] * cb_val_pred
    ens_test = best_w[0] * lgb_test_pred + best_w[1] * xgb_test_pred + best_w[2] * cb_test_pred

    # Bias correction
    bias = y_val_r.mean() / ens_val.mean() if ens_val.mean() > 0 else 1.0
    ens_val_corrected = ens_val * bias
    ens_test_corrected = ens_test * bias

    val_wmape_rest = wmape(y_val_r, ens_val_corrected)

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
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
val_overall = wmape(all_val_preds['quantity'], all_val_preds['predicted'])
print(f"V4 Q4 2024 Val wMAPE:  {val_overall*100:.2f}%  (accuracy: {(1-val_overall)*100:.2f}%)")

test_with_actual = all_test_preds[all_test_preds['quantity'].notna()].copy()
test_overall = wmape(test_with_actual['quantity'], test_with_actual['predicted'])
print(f"V4 Q4 2025 Test wMAPE: {test_overall*100:.2f}%  (accuracy: {(1-test_overall)*100:.2f}%)")

print(f"\n{'='*60}")
print(f"COMPARISON:")
print(f"  V3: Val 12.09% | Test 13.78%")
print(f"  V4: Val {val_overall*100:.2f}% | Test {test_overall*100:.2f}%")
delta_val = (0.1209 - val_overall) * 100
delta_test = (0.1378 - test_overall) * 100
print(f"  Delta: Val {delta_val:+.2f}pp | Test {delta_test:+.2f}pp")
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

# Submission
submission = all_test_preds[['date', 'restaurant_id', 'menu_item_id', 'predicted']].copy()
submission = submission.rename(columns={'predicted': 'predicted_quantity'})
submission['date'] = pd.to_datetime(submission['date']).dt.strftime('%Y-%m-%d')
submission = submission.sort_values(['date','restaurant_id','menu_item_id']).reset_index(drop=True)
assert len(submission) == 69000
submission.to_csv(SUBMISSION, index=False)
print(f"\nSubmission saved: {SUBMISSION} ({len(submission):,} rows)")
print(f"Total pipeline time: {time.time()-t_start:.0f}s")
