*==============================================================================
* setup_and_run_elmps_controlled_income.do
* Main pipeline, income at its measured share (see METHOD). One-file pipeline: locate Python, auto-install the required libraries if
* missing, then run the data-preparation script and the 5 ELMPS imputation
* scripts to produce the .dta outputs. The user never opens a Python prompt.
*
*   Libraries (numpy / pandas / scikit-learn) : left alone if installed at any
*     version; installed at the pinned versions only if missing.
*   Python interpreter itself                 : installed MANUALLY by the user.
*
* If Python is not found, this file prints the download link below and stops.
* Install it, then run this file again.
*
*        DOWNLOAD PYTHON:  https://www.python.org/downloads/
*        During install on Windows, TICK  "Add python.exe to PATH".
*
* SETUP:
*   1. Put this do-file + the six .py files in the SAME folder as the three
*      input .dta files:
*          HIECS_combined_23_8.dta   (all HIECS rounds)
*          ELMPS_combined_23_8.dta   (ELMPS 1998, 2006, 2012, 2018)
*          ELMPS23_inc_pov.dta       (ELMPS 2023)
*   2. File > Change working directory  -> that folder.
*   3. Run this file.
*
* RUN ORDER:
*   0. Data_Prep_Overall.py        reads the three inputs, writes
*                                  HIECS_prepared.dta and ELMPS_prepared.dta
*   1-5. <year>_ELMPS_Imputations_QRF.py   read the two prepared files,
*                                  each writes one imputed ELMPS .dta
*
* Success is judged by whether each script (re)creates its output .dta, not
* by Stata's shell return code, which is unreliable on Windows.
*==============================================================================

clear all
set more off

* python command. Leave as "python"; the script tells you if it must change.
global PY "python"

* the link shown to the user if Python is missing
global PYURL "https://www.python.org/downloads/"

di as txt "Working directory: " c(pwd)

*------------------------------------------------------------------------------
* Detection reads captured output from a temp file, because Stata's shell
* return code is unreliable on Windows.
*------------------------------------------------------------------------------

* ---- 1. is Python present? ----
capture erase "_pycheck.txt"
shell "${PY}" --version > "_pycheck.txt" 2>&1
local pyline ""
capture file close _fh
file open _fh using "_pycheck.txt", read text
file read _fh pyline
file close _fh
capture erase "_pycheck.txt"

if strpos("`pyline'", "Python") == 0 {
    di as error "================================================================"
    di as error " No working Python found."
    di as error " Install Python 3, then run this do-file again:"
    di as error ""
    di as error "     ${PYURL}"
    di as error ""
    di as error " On Windows, TICK 'Add python.exe to PATH' during setup."
    di as error " If Python is installed but not named 'python', set the \$PY"
    di as error " line at the top of this file to its full path."
    di as error "================================================================"
    exit 198
}
di as result "Python found: `pyline'"

* ---- 2. are the libraries present? install the pinned versions if not ----
* A library that is already installed, at any version, is left alone. A library
* that is missing is installed at the version the pipeline was run with
* (numpy 2.4.0, pandas 2.3.3, scikit-learn 1.8.0). Results are reproducible
* to the decimal only with those exact versions; the README states them, and
* the versions found on this machine are printed below for the record.
global NUMPY_V   "2.4.0"
global PANDAS_V  "2.3.3"
global SKLEARN_V "1.8.0"
global PYTHON_V  "3.13"

* small checker script, written here so no quoting problems arise in shell
capture erase "_vercheck.py"
capture file close _fw
file open _fw using "_vercheck.py", write text replace
file write _fw "import sys, importlib.metadata as md" _n
file write _fw "want = {'numpy': '${NUMPY_V}', 'pandas': '${PANDAS_V}', 'scikit-learn': '${SKLEARN_V}'}" _n
file write _fw "missing, found = [], []" _n
file write _fw "for d, v in want.items():" _n
file write _fw "    try:" _n
file write _fw "        found.append(d + ' ' + md.version(d) + (' (pinned)' if md.version(d) == v else ' (pinned ' + v + ')'))" _n
file write _fw "    except md.PackageNotFoundError:" _n
file write _fw "        missing.append(d + '==' + v)" _n
file write _fw "py = f'{sys.version_info.major}.{sys.version_info.minor}'" _n
file write _fw "print('PYVER ' + py)" _n
file write _fw "print('DEPS_OK' if not missing else 'MISSING ' + ' '.join(missing))" _n
file write _fw "print('FOUND ' + '; '.join(found))" _n
file close _fw

capture erase "_depcheck.txt"
shell "${PY}" "_vercheck.py" > "_depcheck.txt" 2>&1
local pyver ""
local depline ""
local foundline ""
capture file close _fh
file open _fh using "_depcheck.txt", read text
file read _fh pyver
file read _fh depline
file read _fh foundline
file close _fh
capture erase "_depcheck.txt"

if strpos("`pyver'", "PYVER ${PYTHON_V}") == 0 {
    di as error "Note: the pipeline was run with Python ${PYTHON_V}; this machine has `pyver'."
    di as error "A different Python minor version is not guaranteed to reproduce"
    di as error "results to the decimal."
}

if strpos("`depline'", "DEPS_OK") == 0 {
    local tolist = subinstr("`depline'", "MISSING ", "", 1)
    di as txt "Missing libraries; installing at the pinned versions: `tolist'"
    shell "${PY}" -m pip install `tolist'

    * re-check after install
    capture erase "_depcheck.txt"
    shell "${PY}" "_vercheck.py" > "_depcheck.txt" 2>&1
    local pyver ""
    local depline ""
    local foundline ""
    capture file close _fh
    file open _fh using "_depcheck.txt", read text
    file read _fh pyver
    file read _fh depline
    file read _fh foundline
    file close _fh
    capture erase "_depcheck.txt"

    if strpos("`depline'", "DEPS_OK") == 0 {
        di as error "Install did not take: `depline'"
        di as error "Common cause: '${PY}' lacks write permission. Run once manually in a terminal:"
        di as error "     ${PY} -m pip install --user numpy==${NUMPY_V} pandas==${PANDAS_V} scikit-learn==${SKLEARN_V}"
        capture erase "_vercheck.py"
        exit 198
    }
}
else {
    di as result "All libraries present; nothing installed."
}
capture erase "_vercheck.py"
local foundlist = subinstr("`foundline'", "FOUND ", "", 1)
di as result "Libraries in use: `foundlist'"
di as txt "Pinned versions for exact reproduction: numpy ${NUMPY_V}, pandas ${PANDAS_V}, scikit-learn ${SKLEARN_V} (see README)."

*------------------------------------------------------------------------------
* 3. confirm the three input files are here
*------------------------------------------------------------------------------
foreach f in "HIECS_combined_23_8.dta" "ELMPS_combined_23_8.dta" "ELMPS23_inc_pov.dta" {
    capture confirm file "`f'"
    if _rc {
        di as error "missing input file: `f'"
        di as error "Put the three input .dta files in this folder, or set the"
        di as error "working directory: File > Change working directory."
        exit 601
    }
}

*------------------------------------------------------------------------------
* 4. run the scripts in order. Success = each output .dta is (re)created.
*    Step 0 writes two files; steps 1-5 write one each.
*------------------------------------------------------------------------------
local py0 "Data_Prep_Overall.py"
local py1 "1998_ELMPS_Imputations_QRF.py"
local py2 "2006_ELMPS_Imputations_QRF.py"
local py3 "2012_ELMPS_Imputations_QRF.py"
local py4 "2018_ELMPS_Imputations_QRF.py"
local py5 "2023_ELMPS_Imputations_QRF.py"

local out0a "HIECS_prepared.dta"
local out0b "ELMPS_prepared.dta"
local out1 "ELMPS_1998_with_imputed_epc_and_povline_qrf.dta"
local out2 "ELMPS_2006_with_imputed_epc_and_povline_qrf_2004_2008.dta"
local out3 "ELMPS_2012_with_imputed_epc_and_povline_qrf.dta"
local out4 "ELMPS_2018_with_imputed_epc_and_povline_qrf.dta"
local out5 "ELMPS_2023_with_imputed_epc_and_povline_qrf.dta"

forvalues i = 0/5 {
    local py "`py`i''"
    di as txt _n "==== running `py' ===="

    capture confirm file "`py'"
    if _rc {
        di as error "missing script: `py'"
        di as error "Put the six .py files in this folder next to the do-file."
        exit 601
    }

    * expected outputs of this step
    if `i' == 0 {
        local outs `""`out0a'" "`out0b'""'
    }
    else {
        local outs `""`out`i''""'
    }

    * remove any stale output so its reappearance proves a fresh success
    foreach out of local outs {
        capture erase "`out'"
    }

    shell "${PY}" "`py'"

    foreach out of local outs {
        capture confirm file "`out'"
        if _rc {
            di as error "`py' did not produce `out'. See the Python error above."
            exit 601
        }
        di as result "ok: `out'"
    }
}

di as result _n "All 6 scripts executed. Prepared and imputed datasets written to " c(pwd) "."
