# ELMPS consumption imputation

This repository imputes per capita household consumption for five waves of the Egypt Labor Market Panel Survey (ELMPS), which has no consumption module, from the Household Income, Expenditure, and Consumption Survey (HIECS), which measures it. For every ELMPS household the imputation produces 100 draws of per capita consumption, with the uncertainty of the imputation carried in the spread of the draws, so that consumption is available in the panel survey for analysis. The waves are 1998, 2006, 2012, 2018, and 2023. Alongside the draws, each wave's output includes a per capita poverty line, built from the HIECS line and brought to that wave's prices, and, for each household, the share of its draws that fall below it.

## Method

Consumption is imputed with a quantile regression forest trained on the HIECS round closest to each ELMPS wave, 1999 for 1998, 2004 and 2008 pooled for 2006, 2012 for 2012, 2017 for 2018, and 2021 for 2023. Each ELMPS household receives 100 draws, each an observed consumption value from HIECS households with similar covariates, with HIECS consumption brought to the wave's prices with the CPI. Household income is used in 2012, 2018, and 2023, the waves where ELMPS collected it, and enters only a share of the forest's trees because the two surveys record it differently. The poverty line is the HIECS household poverty line of the source round, per capita and in the wave's prices. The full method is described in the paper *A machine learning approach to imputing consumption when consumption data is missing: An application to the Egypt Labor Market Panel Survey*.

## Contents

**Data, raw inputs**

| File | Contents |
|---|---|
| `HIECS_combined_23_8.dta` | HIECS household files for all rounds from 1999 to 2021, stacked, with consumption, income, the harmonized covariates, and the household poverty line (`lower_pl`) in the rounds that have one |
| `ELMPS_combined_23_8.dta` | ELMPS household files for 1998, 2006, 2012, and 2018, stacked, with income (2012 and 2018), the same harmonized covariates, and the poverty line of the matching HIECS round |
| `ELMPS23_inc_pov.dta` | The ELMPS 2023 household file, with income, the same covariates, and the HIECS 2021 poverty line |

**Data, prepared (written by `Data_Prep_Overall.py`)**

| File | Contents |
|---|---|
| `HIECS_prepared.dta` | All HIECS rounds after the fixes that apply to every year, in each round's own prices, with per capita consumption added |
| `ELMPS_prepared.dta` | All ELMPS waves after the same fixes, with exactly the HIECS columns except consumption |

**Data, outputs (written by the wave scripts)**

| File | Contents |
|---|---|
| `ELMPS_<year>_with_imputed_epc_and_povline_qrf*.dta` | One file per wave, the ELMPS wave with the imputation columns added (see Output columns below) |

**Code**

| File | Contents |
|---|---|
| `setup_and_run_elmps_controlled_income.do` | Stata runner, runs everything below in order |
| `run_all_controlled_income.py` | The same runner in Python, for users without Stata |
| `Data_Prep_Overall.py` | Builds the two prepared files from the raw inputs |
| `1998_ELMPS_Imputations_QRF.py` to `2023_ELMPS_Imputations_QRF.py` | One script per wave (1998, 2006, 2012, 2018, 2023) |

**Environment**

| File | Contents |
|---|---|
| `requirements.txt` | Pinned library versions |

## How to run

There are two runner files that do the same thing, one for Stata and one for Python. In Stata, set the working directory to this folder and run

```
do setup_and_run_elmps_controlled_income.do
```

Without Stata, open a terminal in this folder and run

```
python run_all_controlled_income.py
```

Both check the inputs and the libraries, run the data preparation, run the five waves, and confirm that each output file was written. They give identical results, since both run the same scripts with the same Python.

To run a single wave by hand, run `Data_Prep_Overall.py` once first, since every wave script reads the prepared files it writes. The waves can then be run in any order. Each wave script writes its output file into this folder. The 2006 output carries the suffix `_2004_2008` for its two source rounds.

## Output columns

Each output file adds these columns to the ELMPS wave.

`epc_mi_001` to `epc_mi_100` are the 100 imputation draws of per capita consumption. `epc_pred` is the forest mean prediction. `epc_mi_mean` and `epc_mi_std` are the mean and standard deviation across draws. `povline_pc` is the per capita poverty line. `prob_poor` is the share of draws below the line, which is the household poverty probability.

A household is imputed only if it has a value for every predictor its wave uses. A household with any predictor missing stays in the output file, with missing values in all the imputation columns. This affects 37 of 4,816 households in 1998, 2 of 8,351 in 2006, 8 of 12,060 in 2012, 1,342 of 15,728 in 2018, and 1,820 of 17,758 in 2023. In 2018 and 2023 these are mostly households with missing income. A household that reports zero income is not treated as missing and is imputed.

Separately, the data preparation removes 44 ELMPS households before any imputation, 40 whose head is recorded as younger than 15 and 4 in 2018 whose head's age is missing. These households do not appear in the output files at all.

## Requirements

Python with NumPy, pandas, and scikit-learn. No other packages are needed.

Both runners check for the three libraries before running anything. A library that is already installed, at any version, is left as it is. A library that is missing is installed at the version listed below. The versions found on the machine are printed at the start of the run.

## Reproducibility

Every wave uses seed 42, so the results are the same on every run given the same prepared inputs and the same library versions. The outputs in this repository were produced with Python 3.13.7 (64-bit) on Windows 11, NumPy 2.4.0, pandas 2.3.3, and scikit-learn 1.8.0.

Results match to the decimal only with these exact versions, because a change in scikit-learn or NumPy can alter how the trees are grown or how the random numbers are drawn. Other versions give the same picture but not necessarily the same digits. The library versions are recorded in `requirements.txt` in this folder. To reproduce exactly on a machine that has different versions, install them with

```
pip install -r requirements.txt
```
