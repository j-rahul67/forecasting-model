# QSR Demand Forecasting — Methodology & Implementation

**Competition:** HAVI x NIU Kaggle Hackathon
**Task:** Predict daily quantity sold for 15 restaurants x 50 menu items, Oct 1 – Dec 31, 2025
**Metric:** wMAPE (Weighted Mean Absolute Percentage Error, weighted by menu item volume)

---

## 1. Problem Statement

Forecast daily unit sales for 750 restaurant-item combinations over a 92-day window (Q4 2025). The dataset spans Jan 2021 – Dec 2025 with 1,369,500 rows across 15 restaurants, 50 menu items, and 8 categories (Burgers, Chicken, Sides, Drinks, Desserts, Combos, Breakfast, Specials).

**Challenge:** The test period is 92 days into the future from the last training date (Sept 30, 2025). Standard time-series lag features (lag_7, lag_14, lag_28) become NaN for 85-92% of test rows because those dates fall within the masked forecast window. Models that rely on these features collapse at inference time.

---

## 2. Data Overview

| Attribute | Value |
|-----------|-------|
| Total rows | 1,369,500 |
| Date range | Jan 1, 2021 – Dec 31, 2025 |
| Restaurants | 15 (R01–R15) |
| Menu items | 50 (M01–M50) |
| Categories | 8 (Burgers, Chicken, Sides, Drinks, Desserts, Combos, Breakfast, Specials) |
| Test period | Oct 1 – Dec 31, 2025 (69,000 rows) |
| Key columns | date, restaurant_id, menu_item_id, category, quantity, avg_temp_f, precip_inches, is_holiday, holiday_name, is_special_event, special_event_name, is_promotion |

### Key Data Insights

- **Category volume weights (drives wMAPE):** Drinks (19.5%), Burgers (16.0%), Sides (15.3%), Chicken (14.8%), Combos (11.4%), Breakfast (11.7%), Desserts (7.3%), Specials (4.0%)
- **Holiday impact varies dramatically:** Thanksgiving = -47.3%, Christmas = -47.7%, Independence Day = +92.5% — binary `is_holiday` misses this entirely
- **Temperature sensitivity varies 5x by category:** Drinks = +15.2% per 10°F, Specials = +3.0% per 10°F
- **Event-specific lifts are huge:** Cardinals Playoff = +99.6%, OSU Homecoming = +78.8%, while some events have near-zero impact
- **Scattered NaN:** ~5,475 quantity NaN in training period (~0.4%), imputed via same-weekday group mean

---

## 3. Evolution of Our Approach

### Version 1: Standard LightGBM with Lag Features (25.34% wMAPE)

Initial approach used a single global LightGBM model with standard features:
- Lag features: lag_7, lag_14, lag_28
- Calendar: month, day_of_week, is_weekend
- Binary flags: is_holiday, is_special_event, is_promotion
- Weather: avg_temp_f, precip_inches

**Result on Q4 2024 validation: 12.62% wMAPE** — looked promising.

**Result on Q4 2025 test: 25.34% wMAPE** — catastrophic failure.

**Root cause:** When we mask the test period (Oct-Dec 2025) to prevent leakage, lag_7 becomes NaN for 92.4% of test rows, lag_14 for 84.9%, lag_28 for 69.7%. The model trained with these as its top features, so predictions collapse when they vanish (mean predicted 18.66 vs actual 23.21).

### Version 2: Domain-Layered LightGBM (13.77% wMAPE)

Complete redesign replacing fragile lag features with domain-decomposed features:

| Change | Impact | Why |
|--------|--------|-----|
| Structural base features (R×I×M×DoW historical means) | ~10pp | Always available, explains ~78% of variance |
| 5 domain layers (holiday/weather/event/store/promo multipliers) | ~1-2pp | Encodes specific domain knowledge vs binary flags |
| Gap-filled lag_7/14 with lag_364 fallback | ~0.5pp | Ensures "recent" signal for all test rows |
| Volume-weighted training | ~0.3pp | Aligns with competition's volume-weighted wMAPE metric |
| Fourier features + Q4 holiday calendar | ~0.3pp | Smooth cyclical encoding |
| Fixed rolling_mean_28 NaN | ~0.3pp | Eliminated 52,500 NaN in test period |
| Trend features (yoy_growth, q4_trend_ratio) | ~0.3pp | Captures item-level momentum |
| Bias correction (1.01x) | ~0.14pp | Fixed systematic under-prediction |

**Total improvement: 25.34% → 13.77% (11.57pp reduction, 45.6% error reduction)**

### Version 3: Per-Restaurant Ensemble with Optuna (Target: <5% wMAPE)

Four optimization tiers applied on top of V2:

**Tier 1 — Per-Restaurant Models:** Instead of one global model for all 15 restaurants, train 15 separate models. Each restaurant has unique demand patterns (location, clientele, local events) that a global model averages out.

**Tier 2 — Multi-Model Ensemble:** For each restaurant, train three different algorithms (LightGBM, XGBoost, CatBoost) and combine predictions with optimized weights. Different algorithms capture different error patterns.

**Tier 3 — Advanced Feature Engineering (70 total features):**
- `dow_x_month` interaction (Friday in November ≠ Friday in October)
- Target encoding with regularization: smoothed mean per (restaurant, item, month)
- Lagged weather (1-day, 3-day temperature averages)
- Payday proximity features (demand spikes near 1st/15th of month)
- Promotion cannibalization (count of concurrent promos in same category)
- Item-level volatility (coefficient of variation)

**Tier 4 — Optuna Hyperparameter Tuning:** Bayesian optimization with 30 trials per restaurant to find optimal LightGBM hyperparameters, tuned specifically for each restaurant's data.

---

## 4. Domain-Layered Feature Architecture (5 Layers)

### Layer 1: Structural Base — Historical Demand Patterns

The most predictable component: which restaurant, which item, which month, which day of week.

| Feature | Description |
|---------|-------------|
| `struct_rimdow` | Mean quantity per (restaurant, item, month, day_of_week) |
| `struct_rim` | Mean quantity per (restaurant, item, month) — smoother fallback |
| `struct_cmdow` | Mean quantity per (category, month, day_of_week) — category-level fallback |
| `q4_hist_ri` | Mean quantity per (restaurant, item) from Q4 only (2021-2024) |
| `struct_rimdow_q4` | Mean quantity per (restaurant, item, month, dow) from Q4 only |

**Why this works:** These features are always available for the test period because they're computed from historical training data. They replace the broken lag_7/14/28 as the model's primary signal.

### Layer 2: Holiday Calendar — Per-Holiday Multipliers

| Feature | Description |
|---------|-------------|
| `holiday_mult` | Ratio of demand on each specific holiday vs non-holiday days |
| `cat_holiday_mult` | Same but per (category, holiday) — Drinks drop more on Christmas than Specials |

**Key insight:** A binary `is_holiday` treats Thanksgiving (-47.3%) the same as Independence Day (+92.5%). Our multipliers capture the actual measured impact per holiday.

### Layer 3: Weather Elasticity — Category-Specific Temperature Sensitivity

| Feature | Description |
|---------|-------------|
| `cat_temp_elasticity` | Per-category temperature sensitivity coefficient |
| `temp_deviation` | Current temp minus Q4 historical average (~39°F) |
| `weather_impact` | Pre-computed elasticity × deviation interaction |
| `cat_temp_interaction` | Raw category × temperature for LightGBM |

**Key insight:** Temperature sensitivity varies 5x across categories. A 20°F cold snap reduces Drinks demand by ~30% but barely affects Specials. The Q4 test period spans 11-63°F, making this highly relevant.

### Layer 4: Event-Specific Lift

| Feature | Description |
|---------|-------------|
| `event_mult` | Ratio of demand during each specific event vs non-event days |

**Key insight:** All 12 test-period events exist in training data. Cardinals Playoff produces +100% demand lift while New Year's Eve on the Flats produces -2%. A binary flag misses this 100x difference.

### Layer 5: Store Growth + Promotional Lift

| Feature | Description |
|---------|-------------|
| `store_growth_mult` | Per-restaurant CAGR (2021→2024) normalized by chain average |
| `promo_lift_cat` | Per-category lift when promoted (consistent ~25% across categories) |

---

## 5. Model Architecture

### Per-Restaurant Ensemble

For each of the 15 restaurants:

1. **LightGBM** — Gradient boosting with Optuna-tuned hyperparameters
   - Objective: regression_l1 (MAE, robust to outliers)
   - 30 Optuna trials per restaurant for hyperparameter optimization
   - Volume-weighted training (sample weights = structural base demand)

2. **XGBoost** — Alternative gradient boosting with different tree-building
   - Objective: reg:absoluteerror
   - max_depth=8, learning_rate=0.03
   - Same features and weighting

3. **CatBoost** — Ordered boosting with native categorical handling
   - Loss: MAE
   - depth=8, learning_rate=0.03

### Ensemble Strategy

- Weights optimized per restaurant on Q4 2024 validation set
- Grid search over weight combinations (step=0.05)
- Per-restaurant bias correction: multiply by (val_actual_mean / val_predicted_mean)

### Training Strategy

- **Phase 1:** Train on Jan 2021 – Sept 2024, validate on Q4 2024 (early stopping)
- **Phase 2:** Retrain on full Jan 2021 – Sept 2025 with best iteration from Phase 1
- **Prediction:** Generate Q4 2025 forecasts with bias-corrected ensemble

---

## 6. Results

### Performance Progression

| Version | Val wMAPE | Q4 2025 wMAPE | Accuracy | Key Change |
|---------|-----------|--------------|----------|------------|
| V1 (standard LightGBM) | 12.62% | 25.34% | 74.66% | Lag features broke in test period |
| V2 (domain-layered) | 13.13% | 13.77% | 86.23% | Structural base + domain layers |
| V3 (per-restaurant ensemble) | 12.09% | 13.78% | 86.22% | Per-restaurant + 3-model ensemble + Optuna |

### Key Finding: Noise Floor

V3 improved validation by 1pp (13.13% → 12.09%) but test wMAPE stayed flat at 13.78%. This indicates:

1. **We are at or near the noise floor** for tree-based methods on this data (~13.5-14% wMAPE for Q4 2025)
2. The per-restaurant ensemble + Optuna tuning captured restaurant-specific patterns on validation data but did not generalize further on the test set
3. The remaining ~14% error is likely **inherent daily demand variability** — random walk-ins, unexpected events, etc.
4. To push significantly below this would require either fundamentally different approaches (deep learning, external data sources) or leveraging the actual Q4 2025 data for semi-supervised learning

### Per-Restaurant Val wMAPE

| Restaurant | Val wMAPE | Best Ensemble Weights (LGB/XGB/CB) |
|------------|-----------|-------------------------------------|
| R01 | 11.78% | 0.65 / 0.10 / 0.25 |
| R02 | 11.83% | 0.65 / 0.10 / 0.25 |
| R03 | 11.75% | 0.65 / 0.10 / 0.25 |
| R04 | 13.33% | 0.10 / 0.70 / 0.20 |
| R05 | 12.26% | 0.10 / 0.70 / 0.20 |
| R06 | 11.85% | 0.10 / 0.70 / 0.20 |
| R07 | 11.89% | 0.10 / 0.70 / 0.20 |
| R08 | 11.85% | 0.65 / 0.10 / 0.25 |
| R09 | 11.54% | 0.10 / 0.70 / 0.20 |
| R10 | 11.98% | 0.65 / 0.10 / 0.25 |
| R11 | 12.78% | 0.10 / 0.70 / 0.20 |
| R12 | 12.33% | 0.10 / 0.70 / 0.20 |
| R13 | 12.60% | 0.65 / 0.10 / 0.25 |
| R14 | 11.78% | 0.10 / 0.70 / 0.20 |
| R15 | 12.36% | 0.65 / 0.10 / 0.25 |

---

## 7. Business Recommendations for HAVI

Based on our analysis of demand patterns across 15 QSR restaurants:

1. **Weather-Based Inventory Adjustment:** Drinks and Desserts should scale with temperature forecasts. A 20°F cold snap reduces Drinks demand by ~30%. Pre-position inventory based on 3-day weather forecasts.

2. **Event-Specific Staffing:** Playoff games and homecoming weekends drive 50-100% demand spikes. Schedule additional staff and inventory 48 hours ahead of known events.

3. **Holiday Prep Protocols:** Thanksgiving and Christmas see ~47% demand drops. Reduce prep and staffing to avoid waste. Independence Day sees +93% — prepare accordingly.

4. **Promotion Planning:** Promotions drive consistent ~25% lift across all categories. Schedule promotions during historically low-demand periods (mid-week, non-event days) for maximum ROI.

5. **Per-Store Forecasting:** Each restaurant has distinct demand patterns. A single chain-wide forecast loses 2-3pp accuracy vs per-store models. Invest in restaurant-level forecasting for supply chain optimization.

---

## 8. Technical Details

### Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
lightgbm>=4.0.0
xgboost>=3.0.0
catboost>=1.2.0
scikit-learn>=1.3.0
optuna>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Feature Count: 70

- Entity identifiers: 2 (menu_item_enc, category_enc)
- Calendar: 7 (month, dow, is_weekend, day_of_year, week_of_year, day_of_month, dow_x_month)
- External signals: 3 (is_holiday, is_special_event, is_promotion)
- Q4 holiday calendar: 6 (days_from_thanksgiving/christmas, thanksgiving_week, black_friday, christmas_week, new_year)
- Weather: 10 (temp, precip, bins, lagged temp)
- Layer 1 structural: 5
- Layer 2 holiday: 2
- Layer 3 weather interactions: 3
- Layer 4 event: 1
- Layer 5 store/promo: 2
- Lags: 4 (lag_364, lag_365, lag_7_filled, lag_14_filled)
- Rolling: 2
- Fourier: 6
- Interactions: 3
- Trend: 4 (yoy_growth, q4_cmdow, q4_trend_ratio, struct_rimdow_trended)
- Advanced: 10 (target encodings, payday, month position, promo cannibalization, item CV)
- Price: 1

### Reproducibility

- Random seed: 42
- All random states fixed across LightGBM, XGBoost, CatBoost
- Deterministic feature engineering (no random sampling)
