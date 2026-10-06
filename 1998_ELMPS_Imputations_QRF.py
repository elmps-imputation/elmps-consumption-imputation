# # ELMPS 1998 consumption imputation
#
# Imputes per capita consumption onto ELMPS 1998 from HIECS 1999 by a quantile regression forest with direct leaf
# sampling. The source consumption is deflated to 1998 prices. Each household receives 100 multiple imputation draws,
# every draw a real observed value. The poverty line is the delivered household line `lower_pl`, per capita, brought
# from 1999 to 1998 prices.
#
# Inputs: `HIECS_prepared.dta`, `ELMPS_prepared.dta` (from the overall prep notebook).
# Output: `ELMPS_1998_with_imputed_epc_and_povline_qrf.dta`.
#
# The first cells do the 1998-specific preparation; everything after that is the old 1998 script.

# ## Imports

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ## Read data
#
# Source is HIECS round 1999, target is ELMPS round 1998.

hiecs = pd.read_stata('HIECS_prepared.dta')
elmps = pd.read_stata('ELMPS_prepared.dta')

src     = hiecs[hiecs['round'] == 1999].copy()
elmps98 = elmps[elmps['round'] == 1998].copy()
print('HIECS 1999', src.shape, '| ELMPS 1998', elmps98.shape)

# ## 1998-specific: income
#
# ELMPS 1998 did not measure income, so `totinc` is blank for every 1998 household. This model runs without income.
#
# **Fix.** Drop `totinc` from both frames.

src     = src.drop(columns=['totinc'])
elmps98 = elmps98.drop(columns=['totinc'])

# ## 1998-specific: assets and counts not collected in 1999 or 1998
#
# `camera`, `player`, `microwave`, `sewing`, `computer`, `satd_rec`, `mixer`, `vacuum`, `nb_illiterate`,
# `share_illiterate` are blank for every household in HIECS 1999 and ELMPS 1998.
#
# **Fix.** Drop them from both frames.

EXTRAS = ['camera', 'player', 'microwave', 'sewing', 'computer', 'satd_rec',
          'mixer', 'vacuum', 'nb_illiterate', 'share_illiterate']
src     = src.drop(columns=EXTRAS)
elmps98 = elmps98.drop(columns=EXTRAS)

# ## 1998-specific: variables left out of this model
#
# None. The per-model drops apply to other pairings (`fan` for 2006, the female band 1 and 2 family for 2018,
# `wash` and `job_cat5` and the sewerage merge for 2023). Governorate and location dummies are in.

# ## 1998-specific: prices
#
# HIECS 1999 consumption is in 1999 pounds and the delivered ELMPS 1998 line is the 1999 line copied over, so both are
# brought to 1998 prices with the CPI (2010 = 100), the same factor the old script used.
#
# **Fix.** `epc` times CPI1998/CPI1999 on the source; per capita line = `lower_pl` / `hhsize` times CPI1998/CPI1999 on the target.

CPI_1998, CPI_1999 = 42.59, 43.90
deflator = CPI_1998 / CPI_1999

src['epc'] = src['epc'] * deflator
elmps98['povline_pc'] = elmps98['lower_pl'] / elmps98['hhsize'] * deflator
print('source median epc in 1998 prices:', round(src['epc'].median()), '| target median line per capita:', round(elmps98['povline_pc'].median()))

# ## Feature set X
#
# Every shared numeric column except identifiers, the weight, the round, the line, and the outcome.
# Source and target rows must be complete on X; the few rows with a missing predictor are not used (source) or not imputed (target).

exclude = {'hhid', 'old_hhid', 'chhid', 'cluster1', 'pnum', 'hweight', 'round', 'lower_pl', 'povline_pc', 'totexp', 'epc'}
X = sorted([c for c in src.columns if c not in exclude and c in elmps98.columns
            and src[c].dtype.kind in 'fiub'])
assert not [c for c in X if src[c].isna().all() or elmps98[c].isna().all()], 'a feature is entirely missing on one side'

src            = src.dropna(subset=X).copy()
elmps98_target = elmps98.dropna(subset=X).copy()
print(len(X), 'features |', len(src), 'source rows |', len(elmps98_target), 'of', len(elmps98), 'target rows complete on X')

# ## QRF imputer

def qrf_direct_impute(X_tr, y_tr, X_pred, n_imp=100, seed=42,
                      n_estimators=500, min_samples_leaf=5):
    """
    Each of n_imp draws: for every prediction row, pick a random tree,
    find that row's leaf, draw one random training obs from the leaf.
    Draws are real observed y values, so they stay positive whenever
    y_tr is. No quantile grid, no interpolation, no clipping.
    Returns (imps, mu): imps is (n_pred, n_imp), mu the forest mean.
    """
    X_tr = np.ascontiguousarray(X_tr); y_tr = np.asarray(y_tr, dtype=float)
    X_pred = np.ascontiguousarray(X_pred)
    rf = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        max_features='sqrt', bootstrap=True, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    train_leaves = rf.apply(X_tr)
    pred_leaves  = rf.apply(X_pred)
    n_pred, n_trees = pred_leaves.shape
    members_flat, start_arr, size_arr = [], [], []
    for t in range(n_trees):
        col = train_leaves[:, t]
        order = np.argsort(col, kind='stable')
        uniq, first, counts = np.unique(col[order], return_index=True,
                                        return_counts=True)
        sa = np.zeros(col.max() + 1, dtype=np.int64)
        za = np.zeros(col.max() + 1, dtype=np.int64)
        sa[uniq] = first; za[uniq] = counts
        members_flat.append(order); start_arr.append(sa); size_arr.append(za)
    rng = np.random.default_rng(seed)
    trees_grid = rng.integers(0, n_trees, size=(n_pred, n_imp))
    u = rng.random(size=(n_pred, n_imp))
    leaves_grid = pred_leaves[np.arange(n_pred)[:, None], trees_grid]
    imps = np.empty((n_pred, n_imp), dtype=float)
    for t in range(n_trees):
        mask = trees_grid == t
        if not mask.any():
            continue
        lf = leaves_grid[mask]
        st = start_arr[t][lf]; sz = size_arr[t][lf]
        off = np.minimum((u[mask] * sz).astype(np.int64), sz - 1)
        imps[mask] = y_tr[members_flat[t][st + off]]
    return imps, rf.predict(X_pred)

# ## Imputation

# ---- 1. source: HIECS 1999, epc already in 1998 prices ----
y_src = src['epc'].values
X_src = src[X].values
X_tgt = elmps98_target[X].values

# ---- 2. QRF imputation ----
imps, mu_tgt = qrf_direct_impute(X_src, y_src, X_tgt, n_imp=100, seed=42)
n_imp = imps.shape[1]

# ---- 3. attach imputation columns ----
elmps98['epc_pred'] = np.nan
elmps98.loc[elmps98_target.index, 'epc_pred'] = mu_tgt
elmps98['epc_mi_mean'] = np.nan
elmps98.loc[elmps98_target.index, 'epc_mi_mean'] = imps.mean(axis=1)
elmps98['epc_mi_std'] = np.nan
elmps98.loc[elmps98_target.index, 'epc_mi_std'] = imps.std(axis=1)
for r in range(n_imp):
    col = f'epc_mi_{r+1:03d}'
    elmps98[col] = np.nan
    elmps98.loc[elmps98_target.index, col] = imps[:, r]
print('imputed', len(elmps98_target), 'households with', n_imp, 'draws each')

# ## Poverty line and prob_poor
#
# The per capita line `povline_pc` was set above from `lower_pl`. `prob_poor` is the share of a household's 100 draws below its line.

mi_cols = sorted([c for c in elmps98.columns if re.match(r'^epc_mi_\d{3}$', c)])
sub   = elmps98.loc[elmps98_target.index]
line  = sub['povline_pc'].values
draws = sub[mi_cols].values
elmps98['prob_poor'] = np.nan
elmps98.loc[elmps98_target.index, 'prob_poor'] = (draws < line[:, None]).mean(axis=1)

# ## Headcount check (population weighted, hweight x hhsize)

w = (sub['hweight'] * sub['hhsize']).values
hc_draws = [np.average(draws[:, r] < line, weights=w) for r in range(n_imp)]
print(f'poverty headcount 1998: {100*np.mean(hc_draws):.2f}% (sd across draws {100*np.std(hc_draws):.2f})')

# ## Save

elmps98.to_stata('ELMPS_1998_with_imputed_epc_and_povline_qrf.dta',
                 write_index=False, version=118)
print('Saved: ELMPS_1998_with_imputed_epc_and_povline_qrf.dta')
