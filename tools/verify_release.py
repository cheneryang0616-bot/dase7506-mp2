"""Verify the frozen release with fresh official processes in an independent copy.

No training or model selection occurs here. Temporary copies use the installed
Python environment and a read-only-by-convention link to the official data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
FROZEN_HASHES = {
    'config.json': '913830e7eb6dfbba979be7b4e86a11782960a4a4a15e8b3f9b120b6c22a6bae3',
    'linear.npz': 'fcea0bd554d10c06f5fbc4f9057c859a97e5d4d5db14d8a8132e6a6ac86f1105',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, cwd, env):
    tick = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            capture_output=True, check=True)
    return {'command': command[1:], 'seconds': time.perf_counter() - tick,
            'stdout': result.stdout, 'stderr': result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/corrections/release_verification.json')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'code/PACKAGE_MANIFEST.json').read_text())
    mismatches = [name for name, expected in manifest['files'].items()
                  if digest(ROOT / name) != expected]
    if mismatches:
        raise AssertionError(f'Official files changed: {mismatches}')
    checkpoint = ROOT / 'release/checkpoint/model_bundle'
    assets = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
              for p in sorted(checkpoint.iterdir()) if p.is_file()}
    if {key: value['sha256'] for key, value in assets.items()} != FROZEN_HASHES:
        raise AssertionError('Frozen checkpoint changed')
    asset_bytes = sum(item['bytes'] for item in assets.values())
    assert asset_bytes < 16 * 2**20
    env = os.environ.copy()
    caps = {name: '4' for name in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    env.update(caps)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.pop('PYTHONPATH', None)
    python = str(Path(sys.executable).absolute())
    data_check = invoke([python, 'check_data.py'], ROOT / 'code', env)
    contract = invoke([python, 'tools/test_contract.py'], ROOT, env)
    with tempfile.TemporaryDirectory(prefix='mp2-release-check-') as temp_name:
        temp = Path(temp_name)
        for folder in ('code', 'method'):
            (temp / folder).mkdir()
            for source in (ROOT / folder).glob('*.py'):
                shutil.copy2(source, temp / folder / source.name)
        (temp / 'code/data').symlink_to(ROOT / 'code/data', target_is_directory=True)
        shutil.copytree(checkpoint, temp / 'checkpoint')
        prediction = invoke([
            python, 'predict.py', '--method', '../method/submission.py',
            '--checkpoint', '../checkpoint', '--split', 'test',
            '--output', '../predictions.json'], temp / 'code', env)
        scoring = invoke([
            python, 'score_predictions.py', '--split', 'test',
            '--predictions', '../predictions.json', '--output', '../score.json'],
            temp / 'code', env)
        scores = json.loads((temp / 'score.json').read_text())
        recorded_score = json.loads((ROOT / 'release/test-score.json').read_text())
        assert all(scores[key] == recorded_score[key] for key in ('correct', 'n', 'accuracy'))
        recorded_pred = json.loads((ROOT / 'runs/test-pred.json').read_text())
        generated_pred = json.loads((temp / 'predictions.json').read_text())
        assert all(recorded_pred[key] == generated_pred[key] for key in ('ids', 'selected'))
    assert all(digest(checkpoint / key) == expected for key, expected in FROZEN_HASHES.items())
    result = {
        'status': 'passed',
        'verification_date': '2026-10-22',
        'official_files_checked': len(manifest['files']),
        'official_file_mismatches': mismatches,
        'checkpoint_files': assets,
        'asset_bytes': asset_bytes,
        'frozen_assets_unchanged': True,
        'score': scores,
        'predictions_identical': True,
        'data_check': data_check,
        'contract_check': contract,
        'fresh_prediction': prediction,
        'independent_scoring': scoring,
        'complete_wall_seconds': prediction['seconds'] + scoring['seconds'],
        'thread_caps': caps,
        'python': sys.version,
        'platform': platform.platform(),
        'environment_scope': 'Independent source directory, same installed Python environment; official data linked.',
        'qualification': 'Local CPU evidence only; course reference Xeon decides runtime eligibility.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'status', 'official_files_checked', 'asset_bytes',
        'frozen_assets_unchanged', 'score', 'predictions_identical',
        'complete_wall_seconds', 'environment_scope')}, indent=2))


if __name__ == '__main__':
    main()
