# # ELMPS 2006 consumption imputation
#
# Imputes per capita consumption onto ELMPS 2006 from the HIECS 2004 and 2008 pool, each year rescaled to 2006 prices,
# by a quantile regression forest with direct leaf sampling and 100 draws. The poverty line is the delivered household
# line `lower_pl` (the 2004 line), per capita, brought to 2006 prices.
#
# Inputs: `HIECS_prepared.dta`, `ELMPS_prepared.dta`. Output: `ELMPS_2006_with_imputed_epc_and_povline_qrf_2004_2008.dta`.
#
# The first cells do the 2006-specific preparation; everything after that is the old script.

# ## Imports

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ## Read data
#
# Source is the HIECS 2004 and 2008 pool, target is ELMPS round 2006.

hiecs = pd.read_stata('HIECS_prepared.dta')
elmps = pd.read_stata('ELMPS_prepared.dta')

src     = hiecs[hiecs['round'].isin([2004, 2008])].copy()
elmps06 = elmps[elmps['round'] == 2006].copy()
print('HIECS 2004+2008', src.shape, src['round'].value_counts().to_dict(), '| ELMPS 2006', elmps06.shape)

# ## 2006-specific: income
#
# ELMPS 2006 did not measure income, so `totinc` is blank for every 2006 household. This model runs without income.
#
# **Fix.** Drop `totinc` from both frames.

src     = src.drop(columns=['totinc'])
elmps06 = elmps06.drop(columns=['totinc'])

# ## 2006-specific: assets and counts not collected in 2004, 2008 or 2006
#
# `camera` exists in HIECS 2004, 2008 and ELMPS 2006, so it stays. `player`, `microwave`, `sewing`, `computer`, `satd_rec`,
# `mixer`, `vacuum`, `nb_illiterate`, `share_illiterate` are blank for every household on at least one side.
#
# **Fix.** Drop those nine from both frames.

EXTRAS = ['player', 'microwave', 'sewing', 'computer', 'satd_rec',
          'mixer', 'vacuum', 'nb_illiterate', 'share_illiterate']
src     = src.drop(columns=EXTRAS)
elmps06 = elmps06.drop(columns=EXTRAS)

# ## 2006-specific: `fan`
#
# In HIECS 2008 only 15 percent of households have `fan` = 1 (83 percent in 2004, 79 percent in ELMPS 2006), and those
# households consume 1.9 times the others. The 2008 question counted a rarer, more expensive item, so the column means
# different things on the two sides of this model.
#
# **Fix.** Leave `fan` out of this model.

src     = src.drop(columns=['fan'])
elmps06 = elmps06.drop(columns=['fan'])

# ## 2006-specific: `nb_univabove`, `share_univabove`
#
# Number of university-educated members and its share are blank for 14 percent of HIECS 2008 households (about 3,200).
# Kept, the model loses those training rows; dropped, it loses the two predictors. Both were run; dropping them gave the
# closer headcount.
#
# **Fix.** Leave both columns out of this model.

src     = src.drop(columns=['nb_univabove', 'share_univabove'])
elmps06 = elmps06.drop(columns=['nb_univabove', 'share_univabove'])

# ## 2006-specific: prices
#
# HIECS 2004 and 2008 consumption are in their own years' pounds and the delivered ELMPS 2006 line is the 2004 line copied over.
# All are brought to 2006 prices with the CPI (2010 = 100), the same factors the old script used.
#
# **Fix.** `epc` times CPI2006/CPI(year) on each source year; per capita line = `lower_pl` / `hhsize` times CPI2006/CPI2004 on the target.

CPI = {2004.0: 55.08, 2006.0: 62.17, 2008.0: 80.42}

src['epc'] = src['epc'] * (CPI[2006.0] / src['round'].map(CPI))
elmps06['povline_pc'] = elmps06['lower_pl'] / elmps06['hhsize'] * (CPI[2006.0] / CPI[2004.0])
print('source median epc in 2006 prices by year:', src.groupby('round')['epc'].median().round(0).to_dict(),
      '| target median line per capita:', round(elmps06['povline_pc'].median()))

# ## Feature set X
#
# Every shared numeric column except identifiers, the weight, the round, the line, and the outcome.
# Source and target rows must be complete on X.

exclude = {'hhid', 'old_hhid', 'chhid', 'cluster1', 'pnum', 'hweight', 'round', 'lower_pl', 'povline_pc', 'totexp', 'epc'}
X = sorted([c for c in src.columns if c not in exclude and c in elmps06.columns
            and src[c].dtype.kind in 'fiub'])
assert not [c for c in X if src[c].isna().all() or elmps06[c].isna().all()], 'a feature is entirely missing on one side'

src            = src.dropna(subset=X).copy()
elmps06_target = elmps06.dropna(subset=X).copy()
print(len(X), 'features |', len(src), 'source rows', src['round'].value_counts().to_dict(),
      '|', len(elmps06_target), 'of', len(elmps06), 'target rows complete on X')

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

# ---- 1. source: HIECS 2004 + 2008, epc already in 2006 prices ----
y_src = src['epc'].values
X_src = src[X].values
X_tgt = elmps06_target[X].values

# ---- 2. QRF imputation ----
imps, mu_tgt = qrf_direct_impute(X_src, y_src, X_tgt, n_imp=100, seed=42)
n_imp = imps.shape[1]

# ---- 3. attach imputation columns ----
elmps06['epc_pred'] = np.nan
elmps06.loc[elmps06_target.index, 'epc_pred'] = mu_tgt
elmps06['epc_mi_mean'] = np.nan
elmps06.loc[elmps06_target.index, 'epc_mi_mean'] = imps.mean(axis=1)
elmps06['epc_mi_std'] = np.nan
elmps06.loc[elmps06_target.index, 'epc_mi_std'] = imps.std(axis=1)
for r in range(n_imp):
    col = f'epc_mi_{r+1:03d}'
    elmps06[col] = np.nan
    elmps06.loc[elmps06_target.index, col] = imps[:, r]
print('imputed', len(elmps06_target), 'households with', n_imp, 'draws each')

# ## Poverty line and prob_poor
#
# The per capita line `povline_pc` was set above from `lower_pl`. `prob_poor` is the share of a household's 100 draws below its line.

mi_cols = sorted([c for c in elmps06.columns if re.match(r'^epc_mi_\d{3}$', c)])
sub   = elmps06.loc[elmps06_target.index]
line  = sub['povline_pc'].values
draws = sub[mi_cols].values
elmps06['prob_poor'] = np.nan
elmps06.loc[elmps06_target.index, 'prob_poor'] = (draws < line[:, None]).mean(axis=1)

# ## Poverty and inequality statistics
#
# Per capita consumption, population weighted (`hweight` x `hhsize`), computed on each of the 100 draws; reported as the
# mean across draws with the standard deviation across draws. HIECS 2004 observed (own `epc` against own `lower_pl`) is the comparator.

def wquant(v, w, q):
    i = np.argsort(v); v, w = v[i], w[i]
    cw = np.cumsum(w); return v[np.searchsorted(cw, q * cw[-1])]

def pov_stats(y, w, line):
    w = w / w.sum(); mu = np.sum(w * y)
    i = np.argsort(y); ys, ws = y[i], w[i]; cw = np.cumsum(ws)
    r = y / mu
    q = {p: wquant(y, w, p) for p in (0.10, 0.25, 0.75, 0.90)}
    return {'Poverty headcount (%)': 100 * np.sum(w * (y < line)),
            'Poverty gap, FGT1 (%)': 100 * np.sum(w * np.clip((line - y) / line, 0, None)),
            'Gini coefficient': np.sum(ws * (2 * cw - ws) * ys) / mu - 1,
            'GE(-1)': 0.5 * (np.sum(w * r ** -1) - 1),
            'GE(0)': np.sum(w * np.log(1 / r)),
            'GE(1)': np.sum(w * r * np.log(r)),
            'GE(2)': 0.5 * (np.sum(w * r ** 2) - 1),
            'P90 / P10': q[0.90] / q[0.10],
            'P75 / P25': q[0.75] / q[0.25]}

w = (sub['hweight'] * sub['hhsize']).values
per_draw = pd.DataFrame([pov_stats(draws[:, r], w, line) for r in range(n_imp)])

h04 = hiecs[hiecs['round'] == 2004].dropna(subset=['lower_pl'])
obs = pov_stats(h04['epc'].values, (h04['hweight'] * h04['hhsize']).values, (h04['lower_pl'] / h04['hhsize']).values)

table = pd.DataFrame({'QRF 2006': [f'{m:.3f} ± {s:.3f}' for m, s in zip(per_draw.mean(), per_draw.std(ddof=0))],
                      'HIECS 2004 observed': [f'{v:.3f}' for v in obs.values()]}, index=per_draw.columns)
print(table.to_string())

# ## Save

out = 'ELMPS_2006_with_imputed_epc_and_povline_qrf_2004_2008.dta'
elmps06.to_stata(out, write_index=False, version=118)
print('Saved:', out)
