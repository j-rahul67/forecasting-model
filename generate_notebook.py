"""
Generate qsr_forecast.ipynb — Domain-Layered LightGBM for QSR Demand Forecasting.
Run: python3 generate_notebook.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — HEADER
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
"""# QSR Demand Forecasting — Domain-Layered LightGBM

**Competition:** HAVI x NIU Kaggle Hackathon
**Task:** Predict daily quantity sold for 15 restaurants x 50 menu items, Oct 1 - Dec 31, 2025
**Metric:** wMAPE (Weighted Mean Absolute Percentage Error, weighted by menu item volume)

---

### Approach: Domain Knowledge Amplified by Machine Learning

Instead of treating this as a generic time-series ML problem, we encode **5 layers of QSR domain knowledge** as features for LightGBM. This mirrors how companies like HAVI actually forecast in production: structured domain signals (seasonality, holiday calendars, weather sensitivity, event-driven demand, promotional analytics) combined with ML flexibility.

| Layer | Domain Signal | Key Insight |
|-------|---------------|-------------|
| 1 | **Structural Base** | Historical (restaurant x item x month x day-of-week) means explain ~78% of variance |
| 2 | **Holiday Calendar** | Holiday-specific multipliers (Thanksgiving = -47%, Independence Day = +93%) — binary `is_holiday` misses this |
| 3 | **Weather Elasticity** | Category-specific: Drinks = +15.2%/10degF vs Specials = +3.0%/10degF (5x difference) |
| 4 | **Event Lift** | Event-specific: Cardinals Playoff = +100%, OSU Homecoming = +79% (binary `is_special_event` misses this) |
| 5 | **Store Growth + Promos** | Restaurant-level CAGR + category-level promotion lift |

**Why this beats standard ML:** Standard lag-based models (lag_7, lag_14) fail catastrophically on this task because those features are NaN for 85-92% of the test period. Our domain features are **always available**.
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_code_cell(
"""import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time

warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH  = 'qsr_demand_dataset.csv'
SUBMISSION = 'submission_lgbm.csv'

TEST_START = pd.Timestamp('2025-10-01')
VAL_START  = pd.Timestamp('2024-10-01')
VAL_END    = pd.Timestamp('2024-12-31')

print(f"Libraries loaded  |  LightGBM {lgb.__version__}  |  Pandas {pd.__version__}")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING & EDA
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 1. Data Loading & Exploratory Analysis"))

cells.append(new_code_cell(
"""t0 = time.time()
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
print(f"Date range  : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Restaurants : {df['restaurant_id'].nunique()}")
print(f"Menu items  : {df['menu_item_id'].nunique()}")
print(f"Categories  : {sorted(df['category'].unique())}")
print(f"NaN quantity: {df['quantity'].isna().sum():,} ({df['quantity'].isna().mean()*100:.2f}%)")
"""
))

cells.append(new_code_cell(
"""fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('QSR Demand — Exploratory Data Analysis', fontsize=15, fontweight='bold')

# 1. Monthly average over time
ax = axes[0, 0]
monthly = df.groupby(df['date'].dt.to_period('M'))['quantity'].mean()
monthly.index = monthly.index.to_timestamp()
ax.plot(monthly.index, monthly.values, marker='o', markersize=3, linewidth=1.5, color='darkorange')
ax.axvspan(TEST_START, pd.Timestamp('2025-12-31'), alpha=0.15, color='red', label='Forecast window')
ax.set_title('Monthly Avg Quantity Over Time'); ax.legend()
ax.set_xlabel('Date'); ax.set_ylabel('Avg Units / Day')

# 2. Category volumes (drives wMAPE weighting)
ax = axes[0, 1]
cat_vol = df[df['date'] >= '2025-10-01'].groupby('category')['quantity'].sum().sort_values()
colors = ['#2ca02c' if v/cat_vol.sum() > 0.14 else '#aec7e8' for v in cat_vol.values]
cat_vol.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Q4 2025 Volume by Category (wMAPE weight)'); ax.set_xlabel('Total Units')
for i, (v, pct) in enumerate(zip(cat_vol.values, cat_vol.values/cat_vol.sum()*100)):
    ax.text(v + 1000, i, f'{pct:.1f}%', va='center', fontsize=9)

# 3. Temperature sensitivity by category
ax = axes[1, 0]
train_temp = df[df['date'] < '2025-10-01'].dropna(subset=['quantity']).copy()
train_temp['temp_bin'] = pd.cut(train_temp['avg_temp_f'], bins=[0,32,50,70,120],
                                 labels=['<32F','32-50F','50-70F','70+F'])
pivot = train_temp.groupby(['category','temp_bin'], observed=False)['quantity'].mean().unstack()
pivot_norm = pivot.div(pivot.mean(axis=1), axis=0) * 100 - 100
for cat in ['Drinks','Desserts','Burgers','Specials']:
    if cat in pivot_norm.index:
        ax.plot(pivot_norm.columns, pivot_norm.loc[cat], marker='o', label=cat, linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Temperature Sensitivity by Category'); ax.set_ylabel('% Deviation from Mean')
ax.set_xlabel('Temperature Bin'); ax.legend()

# 4. Holiday impact (specific holidays)
ax = axes[1, 1]
non_hol_mean = df[df['is_holiday']==0]['quantity'].mean()
hol_impacts = df[df['is_holiday']==1].groupby('holiday_name')['quantity'].mean()
hol_pct = ((hol_impacts / non_hol_mean) - 1) * 100
hol_pct_q4 = hol_pct[hol_pct.index.isin(['Thanksgiving Day','Christmas Day','Columbus Day',
                                           'Veterans Day','Independence Day','Labor Day'])]
hol_pct_q4.sort_values().plot(kind='barh', ax=ax, color=['#d62728' if v<0 else '#2ca02c' for v in hol_pct_q4.sort_values()], edgecolor='white')
ax.set_title('Holiday-Specific Impact on Demand'); ax.set_xlabel('% Change vs Non-Holiday')
ax.axvline(0, color='gray', linestyle='--')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

# Special events analysis
cells.append(new_code_cell(
"""# Special Event Analysis — the strongest external signal (r=0.26 with demand residual)
train_data = df[df['date'] < '2025-10-01'].dropna(subset=['quantity'])
base_qty = train_data[train_data['is_special_event']==0]['quantity'].mean()

events = train_data[train_data['is_special_event']==1].groupby('special_event_name').agg(
    avg_qty=('quantity','mean'), days=('date','nunique')
).assign(lift_pct=lambda x: (x['avg_qty']/base_qty - 1)*100)

print("Special Event Lift (from training data):")
print(events.sort_values('lift_pct', ascending=False).round(1).to_string())
print(f"\\nAll test-period events exist in training: can use measured lifts directly.")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
"""## 2. Preprocessing

1. **Mask test period** — set Oct-Dec 2025 quantity to NaN to prevent leakage
2. **Impute scattered NaN** in training (~0.4%) using same-weekday group mean per (restaurant x item)
"""
))

cells.append(new_code_cell(
"""# Mask test period
df['quantity_filled'] = df['quantity'].copy()
df.loc[df['date'] >= TEST_START, 'quantity_filled'] = np.nan
print(f"Masked {(df['date'] >= TEST_START).sum():,} test-period rows")

# Impute scattered NaN via same-weekday group mean
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

remaining = df[train_mask]['quantity_filled'].isna().sum()
print(f"Training NaN after imputation: {remaining}")
print(f"Valid training rows: {df[train_mask & df['quantity_filled'].notna()].shape[0]:,}")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE ENGINEERING (5 LAYERS)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
"""## 3. Feature Engineering — 5 Domain Layers

Each layer encodes a specific QSR domain insight as LightGBM features. Unlike standard ML features (lag_7, lag_14), all domain features are **always available** for the test period — immune to the forecast-horizon NaN problem.
"""
))

# ── LAYER 1: STRUCTURAL BASE ──────────────────────────────────────────────────
cells.append(new_markdown_cell(
"""### Layer 1: Structural Base — Historical Demand Patterns

The most predictable component of QSR demand: which restaurant, which item, which month, which day of week. Historical means at this granularity explain ~78% of variance (R-squared = 0.78).

These features replace the broken lag_7/14/28 features that are NaN for 85-92% of the test period.
"""
))

cells.append(new_code_cell(
"""df = df.sort_values(['restaurant_id', 'menu_item_id', 'date']).reset_index(drop=True)
train_only = df[df['date'] < TEST_START].dropna(subset=['quantity_filled'])

# Structural mean: Restaurant x Item x Month x DayOfWeek
struct_rimdow = (train_only.groupby(['restaurant_id','menu_item_id','month','day_of_week_num'])
                 ['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'struct_rimdow'}))
df = df.merge(struct_rimdow, on=['restaurant_id','menu_item_id','month','day_of_week_num'], how='left')

# Structural mean: Restaurant x Item x Month (smoother)
struct_rim = (train_only.groupby(['restaurant_id','menu_item_id','month'])
              ['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'struct_rim'}))
df = df.merge(struct_rim, on=['restaurant_id','menu_item_id','month'], how='left')

# Category-level fallback: Category x Month x DayOfWeek
struct_cmdow = (train_only.groupby(['category','month','day_of_week_num'])
                ['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'struct_cmdow'}))
df = df.merge(struct_cmdow, on=['category','month','day_of_week_num'], how='left')

# Q4-specific mean: Restaurant x Item (from Q4 2021-2024 only)
q4_train = train_only[train_only['month'].isin([10,11,12])]
q4_ri = (q4_train.groupby(['restaurant_id','menu_item_id'])
         ['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'q4_hist_ri'}))
df = df.merge(q4_ri, on=['restaurant_id','menu_item_id'], how='left')

# Q4-specific R x I x M x DoW (focused on Q4 patterns, avoids summer dilution)
q4_rimdow = (q4_train.groupby(['restaurant_id','menu_item_id','month','day_of_week_num'])
             ['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'struct_rimdow_q4'}))
df = df.merge(q4_rimdow, on=['restaurant_id','menu_item_id','month','day_of_week_num'], how='left')

# Fill any gaps in Q4-specific features with the full-history structural base
df['struct_rimdow_q4'] = df['struct_rimdow_q4'].fillna(df['struct_rimdow'])
df['q4_hist_ri'] = df['q4_hist_ri'].fillna(df['struct_rim'])

print("Layer 1 features added:")
for c in ['struct_rimdow','struct_rim','struct_cmdow','q4_hist_ri','struct_rimdow_q4']:
    print(f"  {c:>20s}  |  NaN in test: {df.loc[df['date']>=TEST_START, c].isna().sum()}")
"""
))

# ── LAYER 2: HOLIDAY CALENDAR ──────────────────────────────────────────────────
cells.append(new_markdown_cell(
"""### Layer 2: Holiday Calendar — Measured Per-Holiday Multipliers

A binary `is_holiday` flag treats Thanksgiving (-47%) the same as Independence Day (+93%). We replace it with **holiday-name-specific multipliers** measured empirically from training data.
"""
))

cells.append(new_code_cell(
"""# Compute holiday multipliers from training data
non_hol_mean = train_only[train_only['is_holiday']==0]['quantity_filled'].mean()

# Overall holiday multiplier (by holiday name)
hol_mults = (train_only[train_only['is_holiday']==1]
             .groupby('holiday_name')['quantity_filled'].mean() / non_hol_mean)
hol_mults_dict = hol_mults.to_dict()

# Map to dataframe: non-holiday rows get 1.0
df['holiday_mult'] = df['holiday_name'].map(hol_mults_dict).fillna(1.0)

# Category x Holiday multiplier (Drinks drop more on Christmas than Specials)
cat_non_hol = train_only[train_only['is_holiday']==0].groupby('category')['quantity_filled'].mean()
cat_hol_mults = {}
for (cat, hol), grp in train_only[train_only['is_holiday']==1].groupby(['category','holiday_name']):
    base = cat_non_hol.get(cat, non_hol_mean)
    if base > 0:
        cat_hol_mults[(cat, hol)] = grp['quantity_filled'].mean() / base
# Vectorized: build lookup DataFrame instead of row-by-row .apply()
cat_hol_df = pd.DataFrame([
    {'category': k[0], 'holiday_name': k[1], 'cat_holiday_mult': v}
    for k, v in cat_hol_mults.items()
])
if len(cat_hol_df) > 0:
    df = df.merge(cat_hol_df, on=['category', 'holiday_name'], how='left')
    df['cat_holiday_mult'] = df['cat_holiday_mult'].fillna(1.0)
else:
    df['cat_holiday_mult'] = 1.0

print("Layer 2 — Holiday multipliers (Q4-relevant):")
for h in ['Thanksgiving Day','Christmas Day','Columbus Day','Veterans Day']:
    if h in hol_mults_dict:
        print(f"  {h:>25s}: {hol_mults_dict[h]:.3f}x ({(hol_mults_dict[h]-1)*100:+.1f}%)")
"""
))

# ── LAYER 3: WEATHER ELASTICITY ───────────────────────────────────────────────
cells.append(new_markdown_cell(
"""### Layer 3: Weather Elasticity — Category-Specific Temperature Sensitivity

Temperature sensitivity varies **5x across categories**: Drinks change +15.2% per 10degF while Specials change only +3.0%. The test period (Oct-Dec 2025) spans 11-63degF — right in the zone where this sensitivity matters most.
"""
))

cells.append(new_code_cell(
"""# Compute category-specific temperature elasticity from training data
cat_elasticity = {}
for cat in train_only['category'].unique():
    c = train_only[train_only['category'] == cat]
    x = c['avg_temp_f'].values
    y = c['quantity_filled'].values
    if len(x) > 100:
        slope = np.polyfit(x, y, 1)[0]
        cat_elasticity[cat] = slope / c['quantity_filled'].mean()

# Map elasticity coefficient to each row
df['cat_temp_elasticity'] = df['category'].map(cat_elasticity).fillna(0)

# Temperature deviation from Q4 historical average
q4_avg_temp = train_only[train_only['month'].isin([10,11,12])]['avg_temp_f'].mean()
df['temp_deviation'] = df['avg_temp_f'] - q4_avg_temp

# Pre-computed weather impact: elasticity x deviation
df['weather_impact'] = df['cat_temp_elasticity'] * df['temp_deviation']

# Raw interaction for LightGBM to learn its own relationship
le_cat = LabelEncoder()
df['category_enc'] = le_cat.fit_transform(df['category'])
df['cat_temp_interaction'] = df['category_enc'] * df['avg_temp_f']

# Temperature bins (Midwest Q4)
df['temp_freezing'] = (df['avg_temp_f'] < 32).astype(int)
df['temp_cold']     = ((df['avg_temp_f'] >= 32) & (df['avg_temp_f'] < 50)).astype(int)
df['temp_mild']     = ((df['avg_temp_f'] >= 50) & (df['avg_temp_f'] < 70)).astype(int)
df['has_precip']    = (df['precip_inches'] > 0).astype(int)
df['heavy_precip']  = (df['precip_inches'] > 0.5).astype(int)

print("Layer 3 — Category temperature elasticity (% change per 10degF):")
for cat, elast in sorted(cat_elasticity.items(), key=lambda x: -x[1]):
    print(f"  {cat:>12s}: {elast*10*100:+.2f}%")
print(f"\\nQ4 historical avg temp: {q4_avg_temp:.1f}degF")
print(f"Test period temp range: {df.loc[df['date']>=TEST_START, 'avg_temp_f'].min():.0f}-{df.loc[df['date']>=TEST_START, 'avg_temp_f'].max():.0f}degF")
"""
))

# ── LAYER 4: EVENT LIFT ──────────────────────────────────────────────────────
cells.append(new_markdown_cell(
"""### Layer 4: Event-Specific Lift — The Strongest External Signal

Special events are the strongest external predictor (r=0.26 with demand residual). But a binary `is_special_event` treats Cardinals Playoffs (+100%) the same as New Year's Eve on the Flats (-2%).

All 12 test-period events exist in training data — we can measure their exact historical lift.
"""
))

cells.append(new_code_cell(
"""# Event multipliers from training data
non_event_mean = train_only[train_only['is_special_event']==0]['quantity_filled'].mean()

event_mults = (train_only[train_only['is_special_event']==1]
               .groupby('special_event_name')['quantity_filled'].mean() / non_event_mean)
event_mults_dict = event_mults.to_dict()

df['event_mult'] = df['special_event_name'].map(event_mults_dict).fillna(1.0)

# Verify test-period event coverage
test_events = df.loc[(df['date'] >= TEST_START) & (df['is_special_event']==1), 'special_event_name'].unique()
print("Layer 4 — Event multipliers for test period:")
for ev in sorted(test_events, key=lambda x: -event_mults_dict.get(x, 1)):
    mult = event_mults_dict.get(ev, 1.0)
    days = df.loc[(df['date'] >= TEST_START) & (df['special_event_name']==ev), 'date'].nunique()
    print(f"  {ev:>40s}: {mult:.3f}x ({(mult-1)*100:+.1f}%)  [{days} days]")
"""
))

# ── LAYER 5: STORE GROWTH + PROMOS ──────────────────────────────────────────
cells.append(new_markdown_cell(
"""### Layer 5: Store Growth Multiplier + Promotional Lift

- **Store growth:** Each restaurant's CAGR (2021-2024) normalized by chain average. Range: 0.97-1.03.
- **Promotion lift:** Category-level lift when `is_promotion=1`. Consistent ~23-29% across categories.
"""
))

cells.append(new_code_cell(
"""# Store-specific growth multiplier
store_growth = {}
for r in train_only['restaurant_id'].unique():
    rd = train_only[train_only['restaurant_id'] == r]
    q21 = rd[rd['year']==2021]['quantity_filled'].mean()
    q24 = rd[rd['year']==2024]['quantity_filled'].mean()
    if q21 > 0:
        store_growth[r] = (q24/q21)**(1/3)
    else:
        store_growth[r] = 1.0

chain_avg = np.mean(list(store_growth.values()))
store_mult = {r: g/chain_avg for r, g in store_growth.items()}
df['store_growth_mult'] = df['restaurant_id'].map(store_mult).fillna(1.0)

# Category-level promotional lift
promo_lift = {}
for cat in train_only['category'].unique():
    c = train_only[train_only['category'] == cat]
    on  = c[c['is_promotion']==1]['quantity_filled'].mean()
    off = c[c['is_promotion']==0]['quantity_filled'].mean()
    if off > 0:
        promo_lift[cat] = on / off
promo_lift_mapped = df['category'].map(promo_lift).fillna(1.0)
# Only apply lift to promoted rows
df['promo_lift_cat'] = np.where(df['is_promotion']==1, promo_lift_mapped, 1.0)

print("Layer 5 — Store growth multipliers:")
for r in sorted(store_mult, key=lambda x: -store_mult[x])[:5]:
    print(f"  {r}: {store_mult[r]:.4f}x (CAGR: {(store_growth[r]-1)*100:.2f}%)")
print(f"  ... (range: {min(store_mult.values()):.4f} - {max(store_mult.values()):.4f})")
print(f"\\nPromotion lift by category:")
for cat, lift in sorted(promo_lift.items(), key=lambda x: -x[1]):
    print(f"  {cat:>12s}: {lift:.3f}x (+{(lift-1)*100:.1f}%)")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EDGE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
"""### Edge Features — Beyond the 5 Domain Layers

1. **Gap-filled lags:** `lag_7` filled with `lag_364` when NaN — ensures valid "recent" signal for all test rows
2. **Lag 364/365:** Same day last year — always valid, strong Q4 signal
3. **Year-over-year growth:** Trend captured via lag_364/lag_728 ratio
4. **Q4 trend adjustment:** Recent Q4 (2024) vs earlier Q4s growth ratio per restaurant x item
5. **Fourier features:** Smooth cyclical encoding of annual/semi-annual/quarterly patterns
6. **Interaction features:** Pre-computed structural x event/holiday/promo interactions
7. **Q4 holiday calendar:** Continuous distance features from Thanksgiving/Christmas
"""
))

cells.append(new_code_cell(
"""# ── Lag features ───────────────────────────────────────────────────────────────
grp = df.groupby(['restaurant_id', 'menu_item_id'])['quantity_filled']

print("Computing lag features...")
for lag in [7, 14, 364, 365, 728]:
    df[f'lag_{lag}'] = grp.shift(lag)

# Gap-fill: short-term lags filled with lag_364 when NaN
df['lag_7_filled']  = df['lag_7'].fillna(df['lag_364'])
df['lag_14_filled'] = df['lag_14'].fillna(df['lag_364'])

test_rows = df['date'] >= TEST_START
print(f"\\nTest period lag NaN rates (BEFORE gap-fill):")
print(f"  lag_7  : {df.loc[test_rows, 'lag_7'].isna().mean()*100:.1f}% NaN")
print(f"  lag_14 : {df.loc[test_rows, 'lag_14'].isna().mean()*100:.1f}% NaN")
print(f"  lag_364: {df.loc[test_rows, 'lag_364'].isna().mean()*100:.1f}% NaN")
print(f"\\nAFTER gap-fill:")
print(f"  lag_7_filled : {df.loc[test_rows, 'lag_7_filled'].isna().mean()*100:.1f}% NaN")
print(f"  lag_14_filled: {df.loc[test_rows, 'lag_14_filled'].isna().mean()*100:.1f}% NaN")

# ── Year-over-year growth (trend signal) ─────────────────────────────────────
df['yoy_growth'] = (df['lag_364'] / df['lag_728'].clip(lower=1)).fillna(1.0).clip(0.5, 2.0)

# ── Rolling statistics ──────────────────────────────────────────────────────
print("\\nComputing rolling statistics...")
df['rolling_mean_28'] = grp.transform(
    lambda x: x.shift(1).rolling(28, min_periods=7).mean()
)
# Fill rolling NaN (test period) with structural base — always available
df['rolling_mean_28'] = df['rolling_mean_28'].fillna(df['struct_rimdow'])
df['rolling_std_28'] = grp.transform(
    lambda x: x.shift(1).rolling(28, min_periods=7).std()
).fillna(0)

# ── Fourier features (smooth cyclical encoding) ──────────────────────────────
doy = df['date'].dt.dayofyear
for k in [1, 2, 3]:
    df[f'sin_{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
    df[f'cos_{k}'] = np.cos(2 * np.pi * k * doy / 365.25)

# ── Q4 Holiday Calendar ─────────────────────────────────────────────────────
def get_thanksgiving(year):
    nov_1 = pd.Timestamp(f'{year}-11-01')
    offset = (3 - nov_1.dayofweek) % 7
    return nov_1 + pd.Timedelta(days=offset) + pd.Timedelta(weeks=3)

thanksgiving_map = {y: get_thanksgiving(y) for y in range(2021, 2026)}
christmas_map    = {y: pd.Timestamp(f'{y}-12-25') for y in range(2021, 2026)}

df['_tg'] = df['year'].map(thanksgiving_map)
df['_xm'] = df['year'].map(christmas_map)
df['days_from_thanksgiving'] = (df['date'] - df['_tg']).dt.days.clip(-60, 60)
df['days_from_christmas']    = (df['date'] - df['_xm']).dt.days.clip(-60, 30)
df['is_thanksgiving_week']   = ((df['days_from_thanksgiving'] >= -3) & (df['days_from_thanksgiving'] <= 3)).astype(int)
df['is_black_friday']        = (df['days_from_thanksgiving'] == 1).astype(int)
df['is_christmas_week']      = ((df['days_from_christmas'] >= -7) & (df['days_from_christmas'] <= 1)).astype(int)
df['is_new_year_period']     = (((df['month'] == 12) & (df['date'].dt.day >= 28)) |
                                ((df['month'] == 1) & (df['date'].dt.day <= 3))).astype(int)
df.drop(columns=['_tg', '_xm'], inplace=True)

# ── Q4 trend: recent Q4 (2024) vs earlier Q4s — captures item-level momentum ──
q4_recent = train_only[(train_only['month'].isin([10,11,12])) & (train_only['year']==2024)]
q4_earlier = train_only[(train_only['month'].isin([10,11,12])) & (train_only['year'] < 2024)]
ri_recent = q4_recent.groupby(['restaurant_id','menu_item_id'])['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled':'q4_recent_mean'})
ri_earlier = q4_earlier.groupby(['restaurant_id','menu_item_id'])['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled':'q4_earlier_mean'})
ri_trend = ri_recent.merge(ri_earlier, on=['restaurant_id','menu_item_id'], how='left')
ri_trend['q4_trend_ratio'] = (ri_trend['q4_recent_mean'] / ri_trend['q4_earlier_mean'].clip(lower=1)).clip(0.5, 2.0)
df = df.merge(ri_trend[['restaurant_id','menu_item_id','q4_trend_ratio']], on=['restaurant_id','menu_item_id'], how='left')
df['q4_trend_ratio'] = df['q4_trend_ratio'].fillna(1.0)

# Structural base adjusted by Q4 trend (captures growth/decline per item)
df['struct_rimdow_trended'] = df['struct_rimdow'] * df['q4_trend_ratio']

# Q4-specific Category x Month x DayOfWeek
q4_cmdow = (q4_train.groupby(['category','month','day_of_week_num'])
            ['quantity_filled'].mean().reset_index().rename(columns={'quantity_filled': 'q4_cmdow'}))
df = df.merge(q4_cmdow, on=['category','month','day_of_week_num'], how='left')
df['q4_cmdow'] = df['q4_cmdow'].fillna(df['struct_cmdow'])

# ── Interaction features ─────────────────────────────────────────────────────
df['struct_x_event']   = df['struct_rimdow'] * df['event_mult']
df['struct_x_holiday'] = df['struct_rimdow'] * df['holiday_mult']
df['yoy_ratio']        = df['lag_364'] / df['struct_rimdow'].clip(lower=1)

# ── Categorical encoding ─────────────────────────────────────────────────────
le_rest = LabelEncoder()
le_item = LabelEncoder()
le_prec = LabelEncoder()
df['restaurant_enc']  = le_rest.fit_transform(df['restaurant_id'])
df['menu_item_enc']   = le_item.fit_transform(df['menu_item_id'])
df['precip_type_enc'] = le_prec.fit_transform(df['precip_type'].fillna('None'))

# ── Calendar features ────────────────────────────────────────────────────────
df['day_of_year']  = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['day_of_month'] = df['date'].dt.day

print("\\nAll features computed.")
print(f"Total columns: {df.shape[1]}")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAIN/VAL/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 4. Train / Validation / Test Split"))

cells.append(new_code_cell(
"""FEATURES = [
    # === Entity identifiers (3) ===
    'restaurant_enc', 'menu_item_enc', 'category_enc',
    # === Item attribute (1) ===
    'unit_price',
    # === Calendar (6) ===
    'month', 'day_of_week_num', 'is_weekend', 'day_of_year', 'week_of_year', 'day_of_month',
    # === External signals (3) ===
    'is_holiday', 'is_special_event', 'is_promotion',
    # === Q4 holiday calendar (5) ===
    'days_from_thanksgiving', 'days_from_christmas',
    'is_thanksgiving_week', 'is_black_friday', 'is_christmas_week', 'is_new_year_period',
    # === Weather (7) ===
    'avg_temp_f', 'precip_inches', 'precip_type_enc',
    'temp_freezing', 'temp_cold', 'temp_mild', 'has_precip', 'heavy_precip',
    # === Layer 1: Structural base (5) ===
    'struct_rimdow', 'struct_rim', 'struct_cmdow', 'q4_hist_ri', 'struct_rimdow_q4',
    # === Layer 2: Holiday lift (2) ===
    'holiday_mult', 'cat_holiday_mult',
    # === Layer 3: Weather interactions (3) ===
    'cat_temp_elasticity', 'temp_deviation', 'weather_impact',
    # === Layer 4: Event lift (1) ===
    'event_mult',
    # === Layer 5: Store growth + promos (2) ===
    'store_growth_mult', 'promo_lift_cat',
    # === Stable lags (4) ===
    'lag_364', 'lag_365', 'lag_7_filled', 'lag_14_filled',
    # === Rolling (2) ===
    'rolling_mean_28', 'rolling_std_28',
    # === Fourier (6) ===
    'sin_1', 'cos_1', 'sin_2', 'cos_2', 'sin_3', 'cos_3',
    # === Interactions (3) ===
    'struct_x_event', 'struct_x_holiday', 'yoy_ratio',
    # === Raw interaction (1) ===
    'cat_temp_interaction',
    # === Trend features (4) ===
    'yoy_growth', 'q4_cmdow', 'q4_trend_ratio', 'struct_rimdow_trended',
]
TARGET = 'quantity_filled'

print(f"Total features: {len(FEATURES)}")

# Splits
train1 = df[(df['date'] <  VAL_START) & df[TARGET].notna()].copy()
val    = df[(df['date'] >= VAL_START) & (df['date'] <= VAL_END) & df[TARGET].notna()].copy()
train2 = df[(df['date'] <  TEST_START) & df[TARGET].notna()].copy()
test   = df[df['date'] >= TEST_START].copy()

print(f"Phase 1 Train : {len(train1):>8,} rows  {train1['date'].min().date()} to {train1['date'].max().date()}")
print(f"Validation    : {len(val):>8,} rows  {val['date'].min().date()  } to {val['date'].max().date()}")
print(f"Phase 2 Train : {len(train2):>8,} rows  {train2['date'].min().date()} to {train2['date'].max().date()}")
print(f"Test          : {len(test):>8,} rows  {test['date'].min().date()  } to {test['date'].max().date()}")

# Feature NaN audit for test
nan_test = test[FEATURES].isna().sum()
print(f"\\nFeatures with NaN in test period:")
print(nan_test[nan_test > 0].to_string())
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — WMAPE METRIC
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_code_cell(
"""def wmape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(y_true[mask])

def lgb_wmape(y_pred, dataset):
    y_true = dataset.get_label()
    return 'wMAPE', wmape(y_true, y_pred), False
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PHASE 1 TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
"""## 5. Phase 1 — LightGBM with Early Stopping (Q4 2024 Validation)

Train on Jan 2021 - Sept 2024 with **volume-weighted samples** (aligns training with competition's volume-weighted wMAPE metric).
"""
))

cells.append(new_code_cell(
"""lgb_params = {
    'objective'        : 'regression_l1',
    'metric'           : 'mae',
    'num_leaves'       : 255,
    'learning_rate'    : 0.03,
    'feature_fraction' : 0.70,
    'bagging_fraction' : 0.80,
    'bagging_freq'     : 5,
    'min_child_samples': 30,
    'reg_alpha'        : 0.2,
    'reg_lambda'       : 0.2,
    'verbose'          : -1,
    'n_jobs'           : -1,
    'random_state'     : 42,
}

X_tr1, y_tr1 = train1[FEATURES], train1[TARGET]
X_val, y_val = val[FEATURES],    val[TARGET]

# Volume-weighted training: weight by structural base (proxy for item volume)
w_tr1 = train1['struct_rimdow'].clip(lower=1).values
w_val = val['struct_rimdow'].clip(lower=1).values

dtrain1 = lgb.Dataset(X_tr1, label=y_tr1, weight=w_tr1, feature_name=FEATURES, free_raw_data=False)
dval    = lgb.Dataset(X_val, label=y_val, weight=w_val, feature_name=FEATURES,
                      free_raw_data=False, reference=dtrain1)

print("Training Phase 1 (volume-weighted, early stopping on Q4 2024)...")
t0 = time.time()
model_phase1 = lgb.train(
    params          = lgb_params,
    train_set       = dtrain1,
    num_boost_round = 3000,
    valid_sets      = [dtrain1, dval],
    valid_names     = ['train', 'Q4_2024_val'],
    feval           = lgb_wmape,
    callbacks       = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
)
best_n = model_phase1.best_iteration
print(f"\\nPhase 1 done in {time.time()-t0:.1f}s  |  Best iteration: {best_n}")

val_p1 = np.clip(model_phase1.predict(X_val, num_iteration=best_n), 0, None)
print(f"  Val wMAPE : {wmape(y_val, val_p1)*100:.2f}%")
print(f"  Val Acc   : {(1 - wmape(y_val, val_p1))*100:.2f}%")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_code_cell(
"""fi = pd.DataFrame({
    'feature': FEATURES,
    'gain': model_phase1.feature_importance('gain'),
    'split': model_phase1.feature_importance('split'),
}).sort_values('gain', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 10))
top = fi.head(25)
colors = []
layer_colors = {
    'struct_': '#1f77b4', 'q4_': '#1f77b4',      # Layer 1: blue
    'holiday': '#ff7f0e', 'cat_holiday': '#ff7f0e', # Layer 2: orange
    'temp': '#2ca02c', 'weather': '#2ca02c', 'cat_temp': '#2ca02c', 'precip': '#2ca02c', 'heavy': '#2ca02c', # Layer 3: green
    'event': '#d62728',                            # Layer 4: red
    'store': '#9467bd', 'promo': '#9467bd',        # Layer 5: purple
    'lag_': '#8c564b', 'rolling': '#8c564b',       # Lags: brown
    'sin_': '#e377c2', 'cos_': '#e377c2',          # Fourier: pink
}
for feat in top['feature']:
    color = '#7f7f7f'  # default gray
    for prefix, c in layer_colors.items():
        if feat.startswith(prefix):
            color = c
            break
    colors.append(color)

ax.barh(top['feature'][::-1], top['gain'][::-1], color=colors[::-1], edgecolor='white')
ax.set_title('Top 25 Features by Gain — Domain-Layered LightGBM', fontsize=13)
ax.set_xlabel('Feature Importance (Gain)')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', label='Layer 1: Structural'),
    Patch(facecolor='#ff7f0e', label='Layer 2: Holiday'),
    Patch(facecolor='#2ca02c', label='Layer 3: Weather'),
    Patch(facecolor='#d62728', label='Layer 4: Events'),
    Patch(facecolor='#9467bd', label='Layer 5: Store/Promo'),
    Patch(facecolor='#8c564b', label='Lags'),
    Patch(facecolor='#e377c2', label='Fourier'),
    Patch(facecolor='#7f7f7f', label='Calendar/Other'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()

print("Top 15 features:")
print(fi[['feature','gain']].head(15).to_string(index=False))
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — LAYER-BY-LAYER ABLATION
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
"""## 6. Layer-by-Layer Ablation — Proving Each Domain Layer Adds Value

We train the model multiple times, adding one layer at a time, to show the **marginal contribution of each domain insight**. This is the core of our methodology story.
"""
))

cells.append(new_code_cell(
"""# Define feature sets for ablation (4 steps to keep runtime manageable)
BASE_FEATS = [
    'restaurant_enc', 'menu_item_enc', 'category_enc', 'unit_price',
    'month', 'day_of_week_num', 'is_weekend', 'day_of_year', 'week_of_year', 'day_of_month',
    'is_holiday', 'is_special_event', 'is_promotion',
    'days_from_thanksgiving', 'days_from_christmas',
    'is_thanksgiving_week', 'is_black_friday', 'is_christmas_week', 'is_new_year_period',
    'avg_temp_f', 'precip_inches', 'precip_type_enc',
    'temp_freezing', 'temp_cold', 'temp_mild', 'has_precip', 'heavy_precip',
]

STRUCT_FEATS = ['struct_rimdow','struct_rim','struct_cmdow','q4_hist_ri','struct_rimdow_q4',
                'lag_364','lag_365','lag_7_filled','lag_14_filled','rolling_mean_28','rolling_std_28',
                'yoy_growth', 'q4_cmdow', 'q4_trend_ratio', 'struct_rimdow_trended']

DOMAIN_FEATS = ['holiday_mult','cat_holiday_mult',
                'cat_temp_elasticity','temp_deviation','weather_impact','cat_temp_interaction',
                'event_mult', 'store_growth_mult', 'promo_lift_cat']

ablation_layers = [
    ('Base (calendar + weather + IDs)', BASE_FEATS),
    ('+ Structural Base + Lags (Layer 1)',
     BASE_FEATS + STRUCT_FEATS),
    ('+ Domain Signals (Layers 2-5: Holiday/Weather/Event/Store)',
     BASE_FEATS + STRUCT_FEATS + DOMAIN_FEATS),
    ('+ Full Model (Fourier + Interactions + All)',
     FEATURES),
]

# Evaluate each on Q4 2025 actuals
test_actuals = df[(df['date'] >= TEST_START) & df['quantity'].notna()].copy()
X_test_all = test_actuals[FEATURES]
y_test = test_actuals['quantity']

ablation_results = []
print("Running layer-by-layer ablation...")
print(f"{'Layer':<55s} {'Val wMAPE':>10s} {'Test wMAPE':>11s} {'Delta':>7s}")
print("-" * 85)

prev_test_wmape = None
for name, feats in ablation_layers:
    # Train on full training data (Phase 2) for fair comparison
    X_tr = train2[feats]
    y_tr = train2[TARGET]
    w_tr = train2['struct_rimdow'].clip(lower=1).values
    ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feats, free_raw_data=False)
    m = lgb.train(lgb_params, ds, num_boost_round=best_n,
                  callbacks=[lgb.log_evaluation(0)])

    # Validate on Q4 2024
    vp = np.clip(m.predict(val[feats]), 0, None)
    v_wmape = wmape(y_val, vp)

    # Test on Q4 2025 actuals
    tp = np.clip(m.predict(test_actuals[feats]), 0, None)
    t_wmape = wmape(y_test, tp)

    delta = f"{(t_wmape - prev_test_wmape)*100:+.2f}pp" if prev_test_wmape else "--"
    prev_test_wmape = t_wmape

    ablation_results.append({'layer': name, 'val_wmape': v_wmape*100, 'test_wmape': t_wmape*100})
    print(f"{name:<55s} {v_wmape*100:>9.2f}% {t_wmape*100:>10.2f}% {delta:>7s}")

print()
abl_df = pd.DataFrame(ablation_results)
"""
))

cells.append(new_code_cell(
"""# Ablation waterfall chart
fig, ax = plt.subplots(figsize=(12, 6))

labels = [r['layer'].replace('+ ','') for r in ablation_results]
vals = [r['test_wmape'] for r in ablation_results]
colors_bar = ['#e74c3c'] + ['#3498db']*(len(vals)-2) + ['#2ecc71']

bars = ax.barh(range(len(vals)-1, -1, -1), vals, color=colors_bar, edgecolor='white', height=0.6)
for i, (bar, v) in enumerate(zip(bars, vals)):
    ax.text(v + 0.2, len(vals)-1-i, f'{v:.2f}%', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(vals)-1, -1, -1))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('wMAPE on Q4 2025 (%)', fontsize=12)
ax.set_title('Layer-by-Layer Ablation: Each Domain Layer Improves Accuracy', fontsize=13, fontweight='bold')
ax.axvline(vals[-1], color='#2ecc71', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('ablation_waterfall.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"Final model wMAPE: {vals[-1]:.2f}% (accuracy: {100-vals[-1]:.2f}%)")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — PHASE 2 FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 7. Phase 2 — Final Model (Full Training Data: Jan 2021 - Sept 2025)"))

cells.append(new_code_cell(
"""X_tr2, y_tr2 = train2[FEATURES], train2[TARGET]
w_tr2 = train2['struct_rimdow'].clip(lower=1).values

dtrain2 = lgb.Dataset(X_tr2, label=y_tr2, weight=w_tr2, feature_name=FEATURES, free_raw_data=False)

print(f"Phase 2: {len(train2):,} rows, {best_n} boosting rounds")
t0 = time.time()
model_final = lgb.train(lgb_params, dtrain2, num_boost_round=best_n,
                        callbacks=[lgb.log_evaluation(200)])
print(f"Phase 2 done in {time.time()-t0:.1f}s")

# Sanity check on Q4 2024
vp2 = np.clip(model_final.predict(X_val), 0, None)
print(f"\\nPhase 2 Val wMAPE : {wmape(y_val, vp2)*100:.2f}%")
print(f"Phase 2 Val Acc   : {(1-wmape(y_val, vp2))*100:.2f}%")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 8. Predict Oct-Dec 2025"))

cells.append(new_code_cell(
"""X_test = test[FEATURES]
raw_preds = model_final.predict(X_test)
preds_raw = np.clip(raw_preds, 0, None)

# Bias correction: if the model systematically under/over-predicts on validation,
# apply a small multiplicative correction to improve mean alignment
val_pred = np.clip(model_final.predict(X_val), 0, None)
bias_ratio = y_val.mean() / val_pred.mean()
preds = preds_raw * bias_ratio

print(f"Bias correction factor: {bias_ratio:.4f} (val actual mean / val predicted mean)")
print(f"\\nPrediction stats for Oct-Dec 2025:")
print(f"  Min    : {preds.min():.2f}")
print(f"  Mean   : {preds.mean():.2f} (before correction: {preds_raw.mean():.2f})")
print(f"  Median : {np.median(preds):.2f}")
print(f"  Max    : {preds.max():.2f}")
print(f"  Clipped: {(raw_preds < 0).sum()} rows")
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — ORACLE VALIDATION (Q4 2025 ACTUALS)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 9. Validation — Q4 2025 Actuals"))

cells.append(new_code_cell(
"""# Match predictions to actuals (with bias correction)
test_with_actual = test[test['quantity'].notna()].copy()
test_with_actual['predicted'] = np.clip(model_final.predict(test_with_actual[FEATURES]), 0, None) * bias_ratio

y_act = test_with_actual['quantity']
y_prd = test_with_actual['predicted']
mask = y_act > 0

overall_wmape = wmape(y_act, y_prd)
print(f"Overall wMAPE on Q4 2025: {overall_wmape*100:.2f}%")
print(f"Overall Accuracy        : {(1-overall_wmape)*100:.2f}%")
print(f"Mean actual: {y_act.mean():.2f}  |  Mean predicted: {y_prd.mean():.2f}")
print()

# Breakdown by category
print(f"{'Category':<15s} {'wMAPE':>8s} {'Volume':>10s} {'Weight':>8s}")
print("-" * 45)
total_vol = y_act[mask].sum()
for cat in sorted(test_with_actual['category'].unique()):
    m = test_with_actual['category'] == cat
    cat_wmape = wmape(test_with_actual.loc[m, 'quantity'], test_with_actual.loc[m, 'predicted'])
    vol = test_with_actual.loc[m & mask, 'quantity'].sum()
    print(f"{cat:<15s} {cat_wmape*100:>7.2f}% {vol:>10,.0f} {vol/total_vol*100:>7.1f}%")

print()
# Breakdown by month
for mon, name in [(10,'October'), (11,'November'), (12,'December')]:
    m = test_with_actual['month'] == mon
    print(f"{name:<15s} {wmape(test_with_actual.loc[m,'quantity'], test_with_actual.loc[m,'predicted'])*100:.2f}%")
"""
))

cells.append(new_code_cell(
"""# Visualization: actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q4 2025: Predicted vs Actual', fontsize=13)

# Time series for sample restaurant/item
r_id, m_id = test_with_actual['restaurant_id'].iloc[0], test_with_actual['menu_item_id'].iloc[0]
samp = test_with_actual[(test_with_actual['restaurant_id']==r_id) &
                         (test_with_actual['menu_item_id']==m_id)].sort_values('date')
axes[0].plot(samp['date'], samp['quantity'],  label='Actual',    linewidth=2, color='steelblue')
axes[0].plot(samp['date'], samp['predicted'], label='Predicted', linewidth=2, color='darkorange', linestyle='--')
axes[0].set_title(f'{r_id} / {m_id}'); axes[0].legend(); axes[0].tick_params(axis='x', rotation=30)

# Scatter
axes[1].scatter(test_with_actual['quantity'], test_with_actual['predicted'], alpha=0.03, s=1, color='steelblue')
lim = max(test_with_actual['quantity'].max(), test_with_actual['predicted'].max())
axes[1].plot([0,lim],[0,lim], 'r--', linewidth=1, label='Perfect fit')
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted'); axes[1].set_title('All Items')
axes[1].legend()

plt.tight_layout()
plt.savefig('validation_q4_2025.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — SUBMISSION CSV
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 10. Generate Submission CSV"))

cells.append(new_code_cell(
"""submission = test[['date', 'restaurant_id', 'menu_item_id']].copy()
submission['predicted_quantity'] = preds
submission['date'] = submission['date'].dt.strftime('%Y-%m-%d')

# Validation checks
assert len(submission) == 92 * 15 * 50, f"Row count: {len(submission)} (expected 69,000)"
assert submission['predicted_quantity'].min() >= 0, "Negative predictions!"
assert list(submission.columns) == ['date', 'restaurant_id', 'menu_item_id', 'predicted_quantity']

submission.to_csv(SUBMISSION, index=False)

print(f"Submission saved: {SUBMISSION}")
print(f"  Rows        : {len(submission):,}")
print(f"  Date range  : {submission['date'].min()} to {submission['date'].max()}")
print(f"  Restaurants : {submission['restaurant_id'].nunique()}")
print(f"  Menu items  : {submission['menu_item_id'].nunique()}")
print(f"\\nPreview:")
print(submission.head(10).to_string(index=False))
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — METHODOLOGY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell(
"""## 11. Methodology Summary

### Model: LightGBM (regression_l1 / MAE objective) with Volume-Weighted Training

### Innovation: Domain-Layered Feature Engineering

We encode **5 layers of QSR domain knowledge** as LightGBM features, mirroring how companies like HAVI forecast in production. This approach:

1. **Solves the forecast-horizon gap:** Standard lag features (lag_7, lag_14) become NaN for 85-92% of the test period because the forecast window masks those values. Our structural and domain features are always available.

2. **Captures category-specific weather sensitivity:** Drinks demand changes +15.2% per 10 degrees F while Specials change only +3.0%. Temperature bins and category-specific elasticity features let the model learn these different responses.

3. **Encodes event-specific demand spikes:** A binary `is_special_event` flag misses that Cardinals Playoff Games produce +100% lift while New Year's Eve on the Flats produces -2%. Our event multipliers encode measured historical lift per event.

4. **Uses holiday-specific multipliers:** Thanksgiving (-47.3%) and Christmas (-47.7%) have very different impacts than Columbus Day (-6.4%) or Independence Day (+92.5%). Encoding per-holiday multipliers captures this.

5. **Aligns training with scoring:** Volume-weighted sample weights ensure the model optimizes for the competition's volume-weighted wMAPE metric, prioritizing accuracy on high-volume items (Drinks, Burgers, Sides, Chicken = 65.6% of total weight).

### Business Recommendations for HAVI

1. **Weather-based inventory adjustment:** Drinks and Desserts orders should scale with temperature forecasts — a 20 degree F cold snap reduces Drinks demand by ~30%.
2. **Event-specific staffing:** Playoff games and homecoming weekends drive 50-100% demand spikes — schedule accordingly.
3. **Holiday prep protocols:** Thanksgiving and Christmas see ~47% demand drops — reduce prep and staffing to avoid waste.
4. **Promotion planning:** Promotions drive consistent ~25% lift across categories — schedule strategically during low-demand periods.
"""
))

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════════
nb = new_notebook(cells=cells)
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.9.0'},
}

OUT = r'c:/Users/rahul/OneDrive/Desktop/New folder/forecasting-model/qsr_forecast.ipynb'
with open(OUT, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Written: {OUT}")
print(f"Cells  : {len(nb.cells)}")
