"""Fixed evidence benchmark; no downloads. --review compares semantic review.

--cases JSON: question, rows, expected (substrings), forbidden, optional abstain.
For fast-path calibration supply real retrieved rows including rerank_score.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
import answer_rag as rag
from benchmark_cases import CASES
from rag_observability import Trace, CURRENT
from local_storage import file_lock
from start_ollama import executable

CANDIDATES = ('qwen3:1.7b', 'qwen3:4b', 'qwen3.5:4b', 'gemma4:e2b')

def percentile95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None

def ollama_memory():
    try:
        result = subprocess.run([str(executable()), 'ps'], capture_output=True, text=True,
                                timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return str(exc)

def metrics(samples):
    times = [s['seconds'] for s in samples]
    valid = [s['json_valid'] for s in samples if s.get('json_valid') is not None]
    abstentions = [s['abstention_correct'] for s in samples if s['abstention_correct'] is not None]
    evidence = [s['evidence_correct'] for s in samples if s['evidence_correct'] is not None]
    mean = lambda xs: sum(xs)/len(xs) if xs else None
    return {'median_seconds': statistics.median(times), 'p95_seconds': percentile95(times),
            'valid_json_rate': mean(valid), 'json_calls': len(valid),
            'correct_abstention_rate': mean(abstentions), 'evidence_correct_rate': mean(evidence),
            'fast_path_hits': sum(s.get('fast_path_used', False) for s in samples), 'samples': len(samples)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(CANDIDATES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--review', action='store_true')
    parser.add_argument('--fast-threshold', type=float)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/benchmark_models.json'))
    parser.add_argument('--max-median-seconds', type=float)
    parser.add_argument('--save-default', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1: parser.error('--repeats must be positive')
    cases = json.loads(args.cases.read_text(encoding='utf-8')) if args.cases else CASES
    report = {'scope': 'fixed labelled evidence; retrieval excluded', 'repeats': args.repeats,
              'num_ctx': rag.NUM_CTX, 'models': [], 'recommendation': None}
    try: report['ollama_version'] = rag.request_json('/api/version')['version']
    except Exception as exc: report['error'] = str(exc)
    variants = ['standard'] + (['review'] if args.review else []) + (['fast'] if args.fast_threshold is not None else [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with file_lock(Path(__file__).resolve().parent / 'data/operation.lock', timeout=0):
        for model in args.models.split(','):
            model = model.strip()
            try: rag.check_model(model)
            except Exception as exc:
                report['models'].append({'model': model, 'available': False, 'error': str(exc)})
                save()
                continue
            for variant in variants:
                rag.USE_REVIEW = variant == 'review'
                rag.FAST_PATH = variant == 'fast'
                if args.fast_threshold is not None: os.environ['RAG_FAST_MIN_SCORE'] = str(args.fast_threshold)
                samples = []
                entry = {'model': model, 'variant': variant, 'available': True, 'results': samples}
                report['models'].append(entry)
                for repeat in range(args.repeats):
                    for case in cases:
                        trace = Trace(); token = CURRENT.set(trace)
                        start = time.perf_counter()
                        try: result = rag.generate(case['question'], case['rows'], model)
                        except Exception as exc:
                            result = {'status': 'error', 'error': str(exc), 'claims': [], 'json_valid': False}
                        finally: CURRENT.reset(token)
                        elapsed = time.perf_counter() - start
                        quotes = ' '.join(p['quote'] for c in result['claims'] for p in c['evidence'])
                        correct = result['status'] == 'answered' and all(x.casefold() in quotes.casefold() for x in case['expected']) and not any(x.casefold() in quotes.casefold() for x in case['forbidden'])
                        samples.append({'question': case['question'], 'repeat': repeat, 'seconds': elapsed,
                                        'timings': trace.result(), 'status': result['status'],
                                        'json_valid': result.get('json_valid'),
                                        'abstention_correct': result['status'] == 'insufficient' if case.get('abstain') else None,
                                        'evidence_correct': None if case.get('abstain') else correct,
                                        'fast_path_used': result.get('fast_path_used', False),
                                        'quotes': quotes, 'error': result.get('error')})
                        entry.update(metrics(samples))
                        entry['ollama_ps'] = ollama_memory()
                        save()
                        print(model, variant, repeat+1, case['question'], round(elapsed, 3), result['status'], flush=True)
            rag.request_json('/api/generate', {'model': model, 'prompt': '', 'think': False,
                'keep_alive': 0, 'options': {'num_ctx': rag.NUM_CTX}}, timeout=60)
    measured = [r for r in report['models'] if r.get('available') and r.get('variant') == 'standard']
    eligible = [r for r in measured if r.get('valid_json_rate') == 1 and r.get('correct_abstention_rate') == 1 and r.get('evidence_correct_rate') == 1
                and (args.max_median_seconds is None or r['median_seconds'] <= args.max_median_seconds)]
    if eligible: report['best_measured'] = min(eligible, key=lambda r: r['median_seconds'])['model']
    if all(r.get('available') for r in report['models']) and eligible:
        report['recommendation'] = report['best_measured']
        if args.save_default:
            Path('data/model_recommendation.json').write_text(json.dumps({'model': report['recommendation']}), encoding='utf-8')
    report['recommendation_note'] = 'Incomplete comparison or quality failure: retain current default.' if not report['recommendation'] else 'Best standard model meeting all labelled quality checks.'
    save()
    print(json.dumps({k:v for k,v in report.items() if k != 'models'}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
