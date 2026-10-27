"""Real local classifier benchmark; saved contexts, no generation or live video.

Usage: python evaluate_factual_support.py [cases.json]
Each case has id, premise, hypothesis and expected_entailment.
"""
import argparse
import json
from pathlib import Path
import time

from factual_support import classifier
from system_memory import memory_mib

DEFAULT_CASES = Path(__file__).parent / 'benchmarks' / 'factual_support_cases.json'


def free_memory():
    return memory_mib()[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('cases', nargs='?', type=Path, default=DEFAULT_CASES,
                        help='JSON file of test cases (default: the saved benchmark cases)')
    args = parser.parse_args()
    if not args.cases.is_file():
        raise SystemExit(f'Case file not found: {args.cases}. Pass the path to a cases JSON file.')
    root = Path(__file__).parent
    out = root / 'verification_logs' / time.strftime('nli_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    print('Available RAM MiB:', free_memory(), flush=True)
    if free_memory() < 1500:
        raise RuntimeError('Memory gate: need 1500 MiB free before isolated test.')
    results = []
    for case in cases:
        if free_memory() < 256:
            raise RuntimeError('Memory gate: stopped below 256 MiB free.')
        start = time.monotonic()
        decision = classifier.classify(case['premise'], case['hypothesis'])
        row = dict(id=case['id'], expected=case['expected_entailment'],
                   label=decision.label, scores=decision.scores,
                   seconds=round(time.monotonic()-start, 3))
        results.append(row)
        with (out / 'results.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    for threshold in [0.5, 0.8, 0.9]:
        predicted = [r['label'] == 'entailment' and r['scores'].get('entailment', 0) >= threshold for r in results]
        print('Threshold:', threshold, 'correct:', sum(p == r['expected'] for p, r in zip(predicted, results)),
              'of', len(results), 'false accepts:', sum(p and not r['expected'] for p, r in zip(predicted, results)),
              'false rejects:', sum(not p and r['expected'] for p, r in zip(predicted, results)), flush=True)
    print('Results:', out, 'available RAM MiB:', free_memory(), flush=True)
