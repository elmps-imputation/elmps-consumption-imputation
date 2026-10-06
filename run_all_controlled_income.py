# run_all_controlled_income.py
# Main pipeline, income at its measured share (see METHOD). Python equivalent of
# setup_and_run_elmps_controlled_income.do, for users without Stata. Same steps, same order,
# same checks, same scripts, same outputs:
#   0. check that the three input .dta files and the six scripts are in this folder
#   1. check the libraries: one already installed, at any version, is left alone; one that
#      is missing is installed at the pinned version (numpy 2.4.0, pandas 2.3.3,
#      scikit-learn 1.8.0). The versions found are printed for the record.
#   2. run Data_Prep_Overall.py, then the five wave scripts, each as its own process
#      with this same Python, exactly as the do-file does, and stop at the first failure
#   3. confirm every output file was written
#
# Usage, from a terminal opened in this folder:
#     python run_all_controlled_income.py
# Any Python 3 with pip works; results match the published ones to the decimal only with the
# pinned library versions (see README).

import importlib.metadata as md
import os
import subprocess
import sys

PYTHON_V = '3.13'
PINNED = {'numpy': '2.4.0', 'pandas': '2.3.3', 'scikit-learn': '1.8.0'}

INPUTS = ['HIECS_combined_23_8.dta', 'ELMPS_combined_23_8.dta', 'ELMPS23_inc_pov.dta']
STEPS = [  # (script, outputs it must write)
    ('Data_Prep_Overall.py',          ['HIECS_prepared.dta', 'ELMPS_prepared.dta']),
    ('1998_ELMPS_Imputations_QRF.py', ['ELMPS_1998_with_imputed_epc_and_povline_qrf.dta']),
    ('2006_ELMPS_Imputations_QRF.py', ['ELMPS_2006_with_imputed_epc_and_povline_qrf_2004_2008.dta']),
    ('2012_ELMPS_Imputations_QRF.py', ['ELMPS_2012_with_imputed_epc_and_povline_qrf.dta']),
    ('2018_ELMPS_Imputations_QRF.py', ['ELMPS_2018_with_imputed_epc_and_povline_qrf.dta']),
    ('2023_ELMPS_Imputations_QRF.py', ['ELMPS_2023_with_imputed_epc_and_povline_qrf.dta']),
]


def fail(msg):
    print('\nERROR: ' + msg)
    sys.exit(1)


here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
print('Working folder:', here)
print('Python:', sys.version.split()[0], 'at', sys.executable)

# ---- 0. inputs and scripts present? ----
missing = [f for f in INPUTS + [s for s, _ in STEPS] if not os.path.exists(f)]
if missing:
    fail('missing from this folder: ' + ', '.join(missing) +
         '\nPut the three input .dta files and the six .py scripts next to run_all_controlled_income.py.')
print('All input files and scripts found.')

# ---- 1. libraries: leave installed ones alone, install missing ones at the pinned version ----
pyver = f'{sys.version_info.major}.{sys.version_info.minor}'
if pyver != PYTHON_V:
    print(f'Note: the pipeline was run with Python {PYTHON_V}; this machine has {pyver}. '
          f'A different Python minor version is not guaranteed to reproduce results to the decimal.')


def library_status():
    found, absent = [], []
    for dist, want in PINNED.items():
        try:
            have = md.version(dist)
            found.append(f'{dist} {have}' + (' (pinned)' if have == want else f' (pinned {want})'))
        except md.PackageNotFoundError:
            absent.append(f'{dist}=={want}')
    return found, absent


found, absent = library_status()
if absent:
    print('Missing libraries; installing at the pinned versions:', ' '.join(absent))
    r = subprocess.run([sys.executable, '-m', 'pip', 'install'] + absent)
    found, absent = library_status()
    if absent:
        fail('install did not take: ' + ', '.join(absent) +
             '\nCommon cause: no write permission for this Python. Run once manually:\n    ' +
             sys.executable + ' -m pip install --user ' + ' '.join(f'{d}=={v}' for d, v in PINNED.items()))
else:
    print('All libraries present; nothing installed.')
print('Libraries in use:', '; '.join(found))
print('Pinned versions for exact reproduction:', ', '.join(f'{d} {v}' for d, v in PINNED.items()), '(see README)')

# ---- 2. run the scripts in order, each as its own process, stop at the first failure ----
for i, (script, outputs) in enumerate(STEPS):
    print(f'\n---- step {i} of {len(STEPS) - 1}: {script} ----', flush=True)
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        fail(f'{script} exited with code {r.returncode}; see the messages above.')
    not_written = [o for o in outputs if not os.path.exists(o)]
    if not_written:
        fail(f'{script} finished but did not write: ' + ', '.join(not_written))
    print(f'wrote: {", ".join(outputs)}')

# ---- 3. done ----
print(f'\nAll {len(STEPS)} scripts executed. Prepared and imputed datasets written to {here}.')
