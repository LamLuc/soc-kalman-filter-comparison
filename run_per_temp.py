import argparse, json, re, subprocess, sys
from pathlib import Path

# Same split as Section 6.5 of the paper:
#   tune = Mixed1..Mixed7, test = UDDS, LA92, US06, Mixed8
# Only files whose name carries this folder's temperature tag are used
# (guards against a 10degC file sitting in the 0degC folder).
TUNE_PAT = r'_Mixed[1-7]_'
TEST_PAT = r'_(UDDS|LA92|US06|Mixed8)_'


def find_mat(folder, pattern):
    for f in sorted(Path(folder).glob('*.mat')):
        if re.search(pattern, f.name, re.IGNORECASE):
            return f
    return None


def run(cmd):
    str_cmd = [sys.executable] + [str(x) for x in cmd]
    result = subprocess.run(str_cmd, capture_output=False)
    if result.returncode != 0:
        print(f'  ERROR: command returned {result.returncode}')
    return result.returncode


def temp_label(folder_name):
    name = folder_name.lower()
    name = re.sub(r'degc$', '', name)
    name = re.sub(r'c$', '', name)
    name = re.sub(r'^n(\d+)', r'-\1', name)
    return name + 'C'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out-dir', default='results_pertemp')
    ap.add_argument('--soc0', type=float, default=1.0)
    ap.add_argument('--cap-from-run', type=float, default=0.95)
    a = ap.parse_args()

    root = Path(a.root)
    out_dir = Path(a.out_dir)
    tmp_dir = out_dir / 'tmp'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent

    temp_folders = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f'Found {len(temp_folders)} temperature folder(s): {[d.name for d in temp_folders]}')
    all_ok = True

    for folder in temp_folders:
        label = temp_label(folder.name)
        tag = '_' + folder.name.lower() + '_'
        print(f'\n========== Temperature: {label} ({folder.name}) ==========')

        c20 = find_mat(folder, r'c20')
        hppc = find_mat(folder, r'hppc')
        if c20 is None or hppc is None:
            print('  WARNING: missing C20 or HPPC file, skipping.')
            all_ok = False
            continue

        mats = sorted(folder.glob('*.mat'))
        own = [f for f in mats if tag in f.name.lower()]
        tune_files = [f for f in own if re.search(TUNE_PAT, f.name, re.IGNORECASE)]
        test_set = [f for f in own if re.search(TEST_PAT, f.name, re.IGNORECASE)]

        print(f'  C20  : {c20.name}')
        print(f'  HPPC : {hppc.name}')
        print(f'  Tune ({len(tune_files)}): {[f.name for f in tune_files]}')
        print(f'  Test ({len(test_set)}): {[f.name for f in test_set]}')
        if not tune_files or not test_set:
            print('  WARNING: empty tune or test set, skipping.')
            all_ok = False
            continue

        ocv_csv = tmp_dir / f'ocv_{label}.csv'
        params_json = tmp_dir / f'params_{label}.json'
        result_json = out_dir / f'results_{label}.json'

        print('\n  [1/3] Building OCV ...')
        if run([here / 'build_ocv.py', c20, '--out', ocv_csv]) != 0:
            all_ok = False
            continue

        cap = None
        info_path = tmp_dir / f'ocv_{label}_info.json'
        if info_path.exists():
            cap = json.load(open(info_path)).get('Q_discharge_Ah')
        if cap:
            print(f'  Capacity from C20: {cap:.4f} Ah')

        print('\n  [2/3] Fitting HPPC ...')
        cmd = [here / 'fit_hppc.py', hppc, '--ocv', ocv_csv, '--out', params_json]
        if cap:
            cmd += ['--cap', cap]
        if run(cmd) != 0:
            all_ok = False
            continue

        print('\n  [3/3] Running estimators ...')
        cmd = ([here / 'run_real.py', '--ocv', ocv_csv, '--params', params_json, '--tune']
               + tune_files + ['--test'] + test_set
               + ['--soc0', a.soc0, '--cap-from-run', a.cap_from_run, '--out', result_json])
        if cap:
            cmd += ['--cap', cap]
        if run(cmd) != 0:
            all_ok = False
        else:
            print(f'  Results: {result_json}')

    print('\n' + ('=' * 60))
    print('All done.' if all_ok else 'Some temperatures had errors.')


if __name__ == '__main__':
    main()
