"""Summarise historical and corrective experiments without hiding reruns.

Run from any directory. Historical per-row feature times are deliberately not
summed: a feature matrix was reused within each sweep invocation.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def csv_summary(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = Counter(row['run_id'] for row in rows)
    return {
        'path': str(path.relative_to(ROOT)),
        'fit_records': len(rows),
        'unique_run_ids': len(ids),
        'duplicate_run_ids': {key: count for key, count in ids.items() if count > 1},
        'fit_seconds_sum': round(sum(float(row['fit_seconds']) for row in rows), 6),
        'cost_scope': 'Sum of logged fit durations, not end-to-end search wall time.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'release/experiment-summary.json')
    args = parser.parse_args()
    historical = [csv_summary(ROOT / name) for name in (
        'experiments.csv', 'experiments_v1_unstandardized.csv')]
    standalone = []
    for name, description in (
        ('runs/exp_group_relvote.json', 'Historical group loss with gradient discrepancy'),
        ('runs/final_bundle_report.json', 'Original final-model training'),
    ):
        report = json.loads((ROOT / name).read_text())
        standalone.append({'path': name, 'description': description,
                           'fit_seconds': report['fit_seconds']})
    correction_manifest = ROOT / 'runs/corrections/repair_controls_manifest.json'
    corrections = json.loads(correction_manifest.read_text()) if correction_manifest.exists() else None
    correction_reports = []
    for name in ('exp_group_relvote_corrected.json', 'exp_rel_only_C0.002_s42.json',
                 'exp_vote_only_C0.002_s42.json'):
        path = ROOT / 'runs/corrections' / name
        if path.exists():
            report = json.loads(path.read_text())
            correction_reports.append({'path': str(path.relative_to(ROOT)),
                                       'fit_seconds': report['fit_seconds'],
                                       'validation_accuracy_pct': report['validation_accuracy_pct']})
    result = {
        'historical_csv_logs': historical,
        'historical_standalone_fits': standalone,
        'historical_logged_fit_records': sum(x['fit_records'] for x in historical) + len(standalone),
        'historical_fit_seconds_sum': round(
            sum(x['fit_seconds_sum'] for x in historical)
            + sum(x['fit_seconds'] for x in standalone), 6),
        'corrective_controls': corrections,
        'corrective_fit_reports': correction_reports,
        'corrective_fit_seconds_sum': round(sum(x['fit_seconds'] for x in correction_reports), 6),
        'corrective_process_wall_seconds': round(sum(x['wall_seconds'] for x in corrections['runs']), 6) if corrections else None,
        'limitations': [
            'Historical feature_seconds repeats reused work; do not sum it across CSV rows.',
            'Historical invocation-level wall times were not saved and cannot be reconstructed exactly.',
            'Duplicate run IDs append CSV rows but overwrite old per-run JSON/bundle paths.',
            'The legacy unstandardised file is a separate preprocessing version, not directly interchangeable.',
            'Corrective validation controls were run after the test result was known; they did not select or change the frozen submission.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'corrective_controls'}, indent=2))


if __name__ == '__main__':
    main()
