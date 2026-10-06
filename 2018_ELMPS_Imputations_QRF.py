# # ELMPS 2018 consumption imputation
#
# Imputes per capita consumption onto ELMPS 2018 from HIECS 2017, inflated to 2018 prices, by a quantile regression forest
# with direct leaf sampling and 100 draws. The poverty line is the delivered household line `lower_pl` (the 2017 line),
# per capita, brought to 2018 prices. HIECS 2017 observed (35.2 percent) is the comparator, one year before the target.
#
# Income enters a measured share of the forest's trees. The share is the reliability of ELMPS income by the HIECS standard:
# the out-of-bag R² of log income on the covariates in ELMPS over the same in HIECS. Income enters as reported; the level
# table below documents the ELMPS-to-HIECS income gap and changes nothing.
#
# Inputs: `HIECS_prepared.dta`, `ELMPS_prepared.dta`. Output: `ELMPS_2018_with_imputed_epc_and_povline_qrf.dta`.

# ## Imports and settings

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

N_TREES = 500

# ## Read data

hiecs = pd.read_stata('HIECS_prepared.dta')
elmps = pd.read_stata('ELMPS_prepared.dta')
src = hiecs[hiecs['round'] == 2017].copy()
elmps18 = elmps[elmps['round'] == 2018].copy()
print('HIECS 2017', src.shape, '| ELMPS 2018', elmps18.shape)

# ## 2018-specific: income
#
# Both surveys measured income, but not the same way: ELMPS income is lower than HIECS income for the same kind of
# household and includes zeros that HIECS never has. On the 2012 same-year pair a forest that reads income in every tree
# overshot poverty by 2.4 points; on this pair it gave 37.8 against 35.2 for HIECS 2017.
#
# **Fix.** The reliability of ELMPS income, measured on the income columns before any consumption is imputed, sets how
# many trees see income. Households with income missing are not imputed; zero income is a real value and stays.

print('ELMPS 2018 households with income missing:', int(elmps18['totinc'].isna().sum()),
      '| with zero income:', int((elmps18['totinc'] == 0).sum()))

# ## 2018-specific: female band 1 and 2
#
# In HIECS 2017 the female band 1 and 2 columns are copies of the male ones (the copy error also in 2019). ELMPS 2018
# has the correct counts, so the two sides would disagree.
#
# **Fix.** Leave the six female band columns out of this model.

FEMALE = ['female_1', 'female_2', 'nb_female_1', 'nb_female_2', 'share_female_1', 'share_female_2']
src = src.drop(columns=FEMALE)
elmps18 = elmps18.drop(columns=FEMALE)

# ## 2018-specific: sewerage
#
# ELMPS 2018 puts far more households in "no facility" (`sewerge3`) than HIECS 2017 does.
#
# **Fix.** Merge `sewerge2` and `sewerge3` into one "not connected to the network" column.

for df in (src, elmps18):
    df['sewerge_notconn'] = df['sewerge2'] + df['sewerge3']
    df.drop(columns=['sewerge2', 'sewerge3'], inplace=True)

# ## 2018-specific: prices
#
# HIECS 2017 consumption and income are in 2017 pounds, and the delivered ELMPS 2018 line is the HIECS 2017 line copied
# over. All are brought to 2018 prices with the CPI (2010 = 100: 2017: 231.09, 2018: 264.38). ELMPS income is already in
# 2018 pounds.

CPI = {2017.0: 231.09, 2018.0: 264.38}
f = CPI[2018.0] / CPI[2017.0]
src['epc']    = src['epc'] * f
src['totinc'] = src['totinc'] * f
elmps18['povline_pc'] = elmps18['lower_pl'] / elmps18['hhsize'] * f
print('source median epc in 2018 prices:', round(src['epc'].median()), '| target median line per capita:', round(elmps18['povline_pc'].median()))

# ## Feature set X
#
# Every shared numeric column except identifiers, the weight, the round, the line, and the outcome. `X_no` leaves income out,
# `X_inc` adds it. Rows must be complete on `X_inc`.

exclude = {'hhid', 'old_hhid', 'chhid', 'cluster1', 'pnum', 'hweight', 'round', 'lower_pl', 'povline_pc', 'totexp', 'epc', 'totinc'}
X_no  = sorted([c for c in src.columns if c not in exclude and c in elmps18.columns and src[c].dtype.kind in 'fiub'])
X_inc = X_no + ['totinc']
assert not [c for c in X_inc if src[c].isna().all() or elmps18[c].isna().all()], 'a feature is entirely missing on one side'
src = src.dropna(subset=X_inc).copy()
elmps18_target = elmps18.dropna(subset=X_inc).copy()
print(len(X_no), 'covariates |', len(src), 'source rows |', len(elmps18_target), 'of', len(elmps18), 'target rows complete')

# ## Income reliability: how many trees see income
#
# A forest with log income as the outcome and the covariates as inputs, same settings as the imputation forest, fit on HIECS
# and on ELMPS. Out-of-bag R² is the score on households each tree did not see. The share of ELMPS income variation that is
# real signal, by the HIECS standard, is the ELMPS R² over the HIECS R² (the predictable part of income is the same size in
# both surveys and cancels; what is left is true spread over reported spread). That share of the 500 trees sees income.
#
# Income enters as log income, measured on households that reported a positive income in both surveys (HIECS has no
# zeros). A zero is not a noisy report of a number; it is a non-report, and on the log scale it sits ten units below a
# typical household, so a few percent of zeros would dominate the spread and pull the ratio down for a reason unrelated to
# how well reported numbers carry signal. The zeros stay in the imputation as observed values. On the pound scale R² is dominated by the few largest incomes,
# which no covariate predicts and which ELMPS has more of than HIECS; on the 2012 pair the pound scale gave a share of
# 0.19 against 0.42 on the log scale, and only the log-scale share reproduced the known 2012 rate. The trees split on income
# by rank, so the scale changes the R² and not the imputation.

def income_forest(X, y, seed=42):
    rf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=5, max_features='sqrt', bootstrap=True,
                               oob_score=True, random_state=seed, n_jobs=-1)
    rf.fit(np.ascontiguousarray(X), np.asarray(y, dtype=float))
    return rf, rf.oob_score_

pos_s = src['totinc'] > 0
pos_e = elmps18_target['totinc'] > 0
_, r2_h = income_forest(src.loc[pos_s, X_no].values, np.log(src.loc[pos_s, 'totinc'].values))
_, r2_e = income_forest(elmps18_target.loc[pos_e, X_no].values, np.log(elmps18_target.loc[pos_e, 'totinc'].values))
RELIABILITY = min(r2_e / r2_h, 1.0)
print(f'reliability measured on positive incomes: HIECS {int(pos_s.sum())} of {len(src)} | ELMPS {int(pos_e.sum())} of {len(elmps18_target)} '
      f'({100 * (1 - pos_e.mean()):.1f}% zeros)')
rf_h, _ = income_forest(src[X_no].values, src['totinc'].values)     # pound-scale HIECS forest for the level table
N_INC = int(round(N_TREES * RELIABILITY))
print(f'income R² out of bag: HIECS {r2_h:.3f} | ELMPS {r2_e:.3f} | reliability {RELIABILITY:.3f} -> {N_INC} of {N_TREES} trees see income')

# **Result.** Log income on the covariates, out of bag, on households with positive income (HIECS 2017: 12,079; ELMPS 2018: 12,899 of 14,386 (10.3 percent zeros)):
#
# | | R² |
# |---|---|
# | HIECS 2017 | 0.551 |
# | ELMPS 2018 | 0.239 |
# | Reliability = ELMPS 2018 / HIECS 2017 | **0.434** |
#
# So 217 of the 500 trees see income and about 43 percent of draws use it. Headcount from this run: 33.93 ± 0.54 against 35.21 observed in HIECS 2017, one year before.

# ## Income level: the ELMPS-to-HIECS income gap
#
# Documentation of how much less ELMPS households report than HIECS households that look like them. The HIECS income
# forest predicts what each ELMPS household would report if it were in HIECS. Households are cut into deciles of that
# prediction; in each decile the ratio is median reported over median predicted. Nothing here changes the imputation:
# income enters the forest as reported. Rescaling ELMPS income by these ratios was tested on the 2012 same-year pair and
# pushed the headcount below the known rate at every share, so it is not used.

pred = rf_h.predict(np.ascontiguousarray(elmps18_target[X_no].values))
dec = pd.qcut(pred, 10, labels=False, duplicates='drop')
level = pd.DataFrame({'predicted_hiecs_scale': pred, 'reported_elmps': elmps18_target['totinc'].values, 'decile': dec + 1}).groupby('decile').median()
level['ratio'] = level['reported_elmps'] / level['predicted_hiecs_scale']
level['share_zero'] = pd.Series(elmps18_target['totinc'].values == 0).groupby(dec + 1).mean()
print(level.round(3).to_string())
print(f'overall ratio of medians reported / predicted: {np.median(elmps18_target["totinc"].values) / np.median(pred):.3f}')

# ## QRF imputer
#
# Direct leaf sampling in one forest of 500 trees: `N_TREES - N_INC` grown without the income column and `N_INC` with it.
# A draw picks any tree uniformly, so the share of draws that use income equals the reliability. The forest mean is the
# tree-weighted average of the two parts.

def qrf_direct_impute_mixed(X_no_tr, X_inc_tr, y_tr, X_no_pred, X_inc_pred, n_inc, n_imp=100, seed=42,
                            n_estimators=N_TREES, min_samples_leaf=5):
    """
    One forest, two feature sets: n_estimators - n_inc trees grown on X_no (no income) and n_inc on X_inc (with
    income). Each of n_imp draws: for every prediction row, pick one of the trees at random, find the row's leaf
    in it, draw one random training obs from that leaf. Returns (imps, mu, from_income_tree).
    """
    y_tr = np.asarray(y_tr, dtype=float)
    plan = [(X_no_tr, X_no_pred, n_estimators - n_inc, seed), (X_inc_tr, X_inc_pred, n_inc, seed + 1)]
    members_flat, start_arr, size_arr, pred_leaves, means, weights = [], [], [], [], [], []
    for X_tr, X_pred, n, s in plan:
        if n == 0:
            continue
        X_tr = np.ascontiguousarray(X_tr); X_pred = np.ascontiguousarray(X_pred)
        rf = RandomForestRegressor(n_estimators=n, min_samples_leaf=min_samples_leaf,
                                   max_features='sqrt', bootstrap=True, random_state=s, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        train_leaves = rf.apply(X_tr)
        pred_leaves.append(rf.apply(X_pred)); means.append(rf.predict(X_pred)); weights.append(n)
        for t in range(n):
            col = train_leaves[:, t]
            order = np.argsort(col, kind='stable')
            uniq, first, counts = np.unique(col[order], return_index=True, return_counts=True)
            sa = np.zeros(col.max() + 1, dtype=np.int64)
            za = np.zeros(col.max() + 1, dtype=np.int64)
            sa[uniq] = first; za[uniq] = counts
            members_flat.append(order); start_arr.append(sa); size_arr.append(za)
    pred_leaves = np.hstack(pred_leaves)
    n_pred, n_trees = pred_leaves.shape
    n_no = n_estimators - n_inc
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
    mu = sum(w * m for w, m in zip(weights, means)) / sum(weights)
    return imps, mu, trees_grid >= n_no

# ## Imputation

y_src = src['epc'].values
imps, mu_tgt, from_income_tree = qrf_direct_impute_mixed(
    src[X_no].values, src[X_inc].values, y_src,
    elmps18_target[X_no].values, elmps18_target[X_inc].values, n_inc=N_INC, n_imp=100, seed=42)
n_imp = imps.shape[1]

elmps18['epc_pred'] = np.nan
elmps18.loc[elmps18_target.index, 'epc_pred'] = mu_tgt
elmps18['epc_mi_mean'] = np.nan
elmps18.loc[elmps18_target.index, 'epc_mi_mean'] = imps.mean(axis=1)
elmps18['epc_mi_std'] = np.nan
elmps18.loc[elmps18_target.index, 'epc_mi_std'] = imps.std(axis=1)
for r in range(n_imp):
    col = f'epc_mi_{r+1:03d}'
    elmps18[col] = np.nan
    elmps18.loc[elmps18_target.index, col] = imps[:, r]
print('imputed', len(elmps18_target), 'households with', n_imp, 'draws each;', f'{100*from_income_tree.mean():.1f}% of draws from income trees')

# ## Poverty line and prob_poor
#
# `povline_pc` was set above from `lower_pl` in 2018 prices. `prob_poor` is the share of a household's 100 draws below its line.

mi_cols = sorted([c for c in elmps18.columns if re.match(r'^epc_mi_\d{3}$', c)])
sub   = elmps18.loc[elmps18_target.index]
line  = sub['povline_pc'].values
draws = sub[mi_cols].values
elmps18['prob_poor'] = np.nan
elmps18.loc[elmps18_target.index, 'prob_poor'] = (draws < line[:, None]).mean(axis=1)

# ## Poverty and inequality statistics: the check
#
# Population weighted (`hweight` x `hhsize`), computed on each of the 100 draws; reported as the mean across draws with the
# standard deviation across draws. HIECS 2017 observed (own `epc` against own `lower_pl`, in its own prices) is the
# comparator, one year before the target.

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
comp = hiecs[hiecs['round'] == 2017].dropna(subset=['lower_pl'])
obs = pov_stats(comp['epc'].values, (comp['hweight'] * comp['hhsize']).values, (comp['lower_pl'] / comp['hhsize']).values)
table = pd.DataFrame({f'QRF 2018 ({N_INC} income trees, reported income)':
                          [f'{m:.3f} ± {s:.3f}' for m, s in zip(per_draw.mean(), per_draw.std(ddof=0))],
                      'HIECS 2017 observed': [f'{v:.3f}' for v in obs.values()]}, index=per_draw.columns)
print(table.to_string())

# ## Save

out = 'ELMPS_2018_with_imputed_epc_and_povline_qrf.dta'
elmps18.to_stata(out, write_index=False, version=118)
print('Saved:', out)
