# # Data preparation for the ELMPS consumption imputation (overall fixes)
#
# This notebook applies the fixes that hold across all years, to HIECS, ELMPS, or both.
# Everything year-specific stays out of this file and is done in the per-year modeling files:
# CPI adjustment, which waves use income, variables left out of one model, round-specific extra assets,
# and the HIECS 2019 problems.
#
# Inputs: `HIECS_combined_23_8.dta`, `ELMPS_combined_23_8.dta` (1998 to 2018), `ELMPS23_inc_pov.dta` (2023).
#
# Outputs: `HIECS_prepared.dta` and `ELMPS_prepared.dta`, both in each survey's own-year prices,
# with identical column names and order (HIECS has two extra columns: `totexp` and `epc`).

import pandas as pd
import numpy as np

hiecs = pd.read_stata('HIECS_combined_23_8.dta')
elmps_old = pd.read_stata('ELMPS_combined_23_8.dta')
elmps_23 = pd.read_stata('ELMPS23_inc_pov.dta')

# household IDs are numbers in one file and text in the other; make them text before stacking
for df in (elmps_old, elmps_23):
    df['hhid'] = df['hhid'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['old_hhid'] = df['old_hhid'].astype(str).str.replace(r'\.0$', '', regex=True)
elmps = pd.concat([elmps_old, elmps_23], ignore_index=True)

print('HIECS', hiecs.shape, '| ELMPS', elmps.shape)

# ## HIECS: rows with missing consumption
#
# **Issue.** HIECS 2012 contains 7,653 rows that carry `totinc` and nothing else: no consumption,
# no covariates, no weight. They are leftovers from the income merge.
#
# **Fix.** Drop every HIECS row with `totexp` missing.

hiecs = hiecs[hiecs['totexp'].notna()].copy()
print(hiecs.groupby('round').size())

# ## HIECS: Frontier households
#
# **Issue.** Households in the frontier governorates (Red Sea, New Valley, Matrouh, South Sinai) exist only in
# HIECS 2019 and 2021. ELMPS has none, and in 2019 they also lack the urban flag.
#
# **Fix.** Drop them. 2019 has no governorate column, so its frontier rows are identified by the delivered
# region label (correct in 2019). In all other years the label is not used, because it is wrong in 2008, 2010, 2015.

FRONTIER = ['Red Sea', 'El-Wadi El-Gidid', 'Matrouh', 'South Sinai', 'North Sinai']

gov = hiecs['reg'].astype('string').str.strip()
is_frontier = gov.isin(FRONTIER) | (gov.isna() & (hiecs['region'].astype('string') == 'Frontier Governorates'))
hiecs = hiecs[~is_frontier.fillna(False)].copy()
print('dropped', int(is_frontier.sum()), 'frontier rows |', hiecs.shape)

# ## HIECS: per capita consumption
#
# Not in the data. `totexp` is the household total; the model's target is consumption per person.
#
# **Fix.** Compute `epc = totexp / hhsize`.

hiecs['epc'] = hiecs['totexp'] / hiecs['hhsize']
print(hiecs.groupby('round')['epc'].median().round(0))

# ## ELMPS: head age under 15
#
# **Issue.** 40 households have a head aged 0 to 14 (14 in 2018, 26 in 2023). A child cannot be a household head;
# these are entry errors. HIECS heads start at 12.
#
# **Fix.** Drop them. The filter also drops the 4 households in 2018 whose head age is missing.

elmps = elmps[elmps['age'] >= 15].copy()
print(elmps.groupby('round').size())

# ## ELMPS: empty consumption column
#
# `totexp` exists in the ELMPS files but is blank for every household (ELMPS has no consumption; that is what gets imputed).
#
# **Fix.** Drop it, so the only HIECS-only columns are `totexp` and `epc`.

elmps = elmps.drop(columns=['totexp'])

# ## ELMPS: missing income
#
# **Decision.** Households with `totinc` missing are dropped from imputation, like any household with a missing predictor.
# Zero income is a real value and stays.
#
# Not applied here: ELMPS 1998 and 2006 have no income at all, so the drop belongs in the 2012, 2018 and 2023 files.

# ## Region (both surveys)
#
# **Issue.** Region is stored as a number 1 to 6 with a label dictionary attached. In HIECS 2008, 2010 and 2015 the
# numbers follow a different scheme, so the labels are wrong (for example, 2008 "Lower Egypt Rural" is really
# Cairo, Alexandria, Port Said, Suez). ELMPS 2018 and 2023 also put Ismailia in Upper Egypt while every other round
# puts it in Lower Egypt.
#
# **Fix.** Throw away the labels and rebuild region from governorate plus urban flag with one rule, in both surveys,
# every year, then one-hot into five columns. HIECS 2019 has no governorate column; its delivered labels are correct
# and are kept.

URBAN_GOV = ['Cairo', 'Alex.', 'Port-Said', 'Suez']
LOWER = ['Damietta', 'Dakahlia', 'Sharkia', 'Kalyoubia', 'Kafr-Elsheikh', 'Gharbia', 'Menoufia', 'Behera', 'Ismailia']
UPPER = ['Giza', 'Beni-Suef', 'Fayoum', 'Menia', 'Asyout', 'Suhag', 'Qena', 'Aswan', 'Luxur']
GOV_FIX = {'5.0': 'Damietta', '6.0': 'Dakahlia'}   # HIECS 2010 stores these two governorates as codes
LABEL_TO_KEY = {'Urban Governorates': 'urban_gov', 'Lower Egypt Urban': 'lower_urban', 'Lower Egypt Rural': 'lower_rural',
                'Upper Egypt Urban': 'upper_urban', 'Upper Egypt Rural': 'upper_rural'}
REGIONS = list(LABEL_TO_KEY.values())

def slug(s):
    return ''.join(ch if ch.isalnum() else '_' for ch in str(s).strip().lower()).strip('_').replace('__', '_')

def one_hot(df, col, categories, prefix):
    for c in categories:
        df[f'{prefix}_{slug(c)}'] = (df[col] == c).where(df[col].notna()).astype('float32')   # missing stays missing

def region_key(gov, urban, label):
    if pd.isna(gov):                       # HIECS 2019 only: no governorate, label is correct
        return LABEL_TO_KEY.get(label)
    if gov in URBAN_GOV:
        return 'urban_gov'
    if gov in LOWER:
        return 'lower_' + urban
    if gov in UPPER:
        return 'upper_' + urban
    return None

for df in (hiecs, elmps):
    df['gov'] = df['reg'].astype('string').str.strip().replace(GOV_FIX)
    df['urb'] = df['urban'].astype('string')
    df['region'] = [region_key(g, u, l) for g, u, l in zip(df['gov'], df['urb'], df['region'].astype('string'))]
    one_hot(df, 'region', REGIONS, 'region')
    print(df['region'].isna().sum(), 'rows without region')

# ## Governorate (both surveys)
#
# **Issue.** The delivered yes/no columns `reg1` to `reg29` are blank where they should be 0
# (a governorate with no households that year gets a blank column instead of zeros). The text column `reg` is complete.
#
# **Fix.** One-hot encode from the text column, 22 governorates (the ELMPS set). The delivered dummies are dropped later.
# HIECS 2019 has no governorate, so its 22 columns are missing there (handled in the 2023 file).

GOVS = URBAN_GOV + LOWER + UPPER
for df in (hiecs, elmps):
    one_hot(df, 'gov', GOVS, 'gov')
print([c for c in hiecs.columns if c.startswith('gov_')])

# ## Location (both surveys)
#
# **Issue.** Same blank-instead-of-0 problem in `location1` to `location47`, which are governorate-by-urban/rural cells.
#
# **Fix.** One-hot encode 40 cells built from governorate plus urban flag, the same inputs as region.
# The four urban governorates are urban only.

LOCS = [g + ' urban' for g in URBAN_GOV] + [g + ' ' + u for g in LOWER + UPPER for u in ('urban', 'rural')]
for df in (hiecs, elmps):
    df['loc'] = df['gov'] + ' ' + df['urb']
    one_hot(df, 'loc', LOCS, 'loc')
print(len(LOCS), 'location columns')

# ## Marital status (both surveys)
#
# Text column with the same four values on both sides (married, single, widowed, divorced/separated), no dummies yet.
# Missing for 43 households in ELMPS 2018, 5 in ELMPS 2023 and 4 in HIECS 2010; those stay missing in the dummies and are dropped by the
# model's missing-predictor rule like any other missing value.
#
# **Fix.** One-hot into four columns.

MARITAL = ['married', 'single', 'widowed', 'divorced/seperated']
for df in (hiecs, elmps):
    one_hot(df, 'marital_status', MARITAL, 'mar')
print([c for c in hiecs.columns if c.startswith('mar_')])

# ## Head age group (both surveys)
#
# **Fix.** Add three yes/no columns from the head's age: `head_child` (under 18), `head_adult` (18 to 64), `head_retired` (65 and over).
# `age` and `age_sq` stay in.

for df in (hiecs, elmps):
    df['head_child'] = (df['age'] < 18).astype('int8')
    df['head_adult'] = ((df['age'] >= 18) & (df['age'] < 65)).astype('int8')
    df['head_retired'] = (df['age'] >= 65).astype('int8')
print(hiecs[['head_child', 'head_adult', 'head_retired']].mean().round(3).to_dict())

# ## Household size squared (both surveys)
#
# Not in the data. Computed in int64 so large households do not overflow.

for df in (hiecs, elmps):
    df['hhsize_sq'] = (df['hhsize'].astype('int64') ** 2).astype('int32')

# ## Weights (both surveys)
#
# `expan_hh` is the weight on both sides but on different scales: HIECS weights average 1, ELMPS weights are
# expansion factors (about 1,500 households each). Never pooled. The weight never enters the model; the ELMPS weight
# is used only for the weighted poverty rate.
#
# **Fix.** Rename to `hweight`, the name the modeling code expects.

hiecs = hiecs.rename(columns={'expan_hh': 'hweight'})
elmps = elmps.rename(columns={'expan_hh': 'hweight'})
print('HIECS mean weight', round(hiecs['hweight'].mean(), 2), '| ELMPS mean weight', round(elmps['hweight'].mean(), 0))

# ## Internet (both surveys)
#
# **Issue.** ELMPS kept only the "yes" answers and left the rest blank, so a blank cannot be read as "no".
#
# **Fix.** Drop `internet` from both surveys.

hiecs = hiecs.drop(columns=['internet'])
elmps = elmps.drop(columns=['internet'])

# ## Job categories 5 and 7 (both surveys)
#
# The two surveys sort some jobs into different categories (category 5 twice as common in HIECS, category 7 two to three
# times more common in ELMPS from 2018 on). Kept as they are; nothing to do without the category names.

# ## Columns dropped outright (both surveys)
#
# One-side-only columns, text columns that duplicate dummies we already have, the delivered governorate and location
# dummies replaced above, and the helper text columns used to build region, governorate and location.
# ID columns (`hhid`, `old_hhid`, `chhid`, `cluster1`, `pnum`) are kept but never modeled.

DROP = ['region2', 'male', 'female', 'educexp', 'educabr', 'education_source', 'wat5', 'sewerge',
        'psex', 'h_ownership', 'sewerge_facility', 'wat', 'educ', 'educlvl', 'educlvl1', 'educlvl2',
        'urban', 'region', 'reg', 'location', 'marital_status', 'gov', 'urb', 'loc']
DROP += [f'reg{i}' for i in range(1, 30)] + [f'location{i}' for i in range(1, 48)]
hiecs = hiecs.drop(columns=[c for c in DROP if c in hiecs.columns])
elmps = elmps.drop(columns=[c for c in DROP if c in elmps.columns])
print('HIECS', hiecs.shape, '| ELMPS', elmps.shape)

# ## Column names and order
#
# ELMPS must have exactly the HIECS columns, in the same order, except the two HIECS-only columns `totexp` and `epc`.

HIECS_ONLY = ['totexp', 'epc']
only_in_hiecs = [c for c in hiecs.columns if c not in elmps.columns]
only_in_elmps = [c for c in elmps.columns if c not in hiecs.columns]
assert only_in_hiecs == HIECS_ONLY and only_in_elmps == [], (only_in_hiecs, only_in_elmps)

shared = [c for c in hiecs.columns if c not in HIECS_ONLY]
hiecs = hiecs[shared + HIECS_ONLY]
elmps = elmps[shared]
assert list(elmps.columns) == list(hiecs.columns[:-2])
print(len(shared), 'shared columns, same order on both sides')

# ## Save
#
# Both files are in each survey's own-year prices. The CPI step is done per pairing in the modeling files.

hiecs.to_stata('HIECS_prepared.dta', write_index=False, version=118)
elmps.to_stata('ELMPS_prepared.dta', write_index=False, version=118)
print('HIECS_prepared.dta', hiecs.shape, '| ELMPS_prepared.dta', elmps.shape)
print(hiecs.groupby('round').size().to_dict())
print(elmps.groupby('round').size().to_dict())
