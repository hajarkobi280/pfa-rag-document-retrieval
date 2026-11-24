import ast
import json
import os
from pathlib import Path
import re
import threading
import time
import pytest
import app as web
import answer_rag as rag
from benchmark_cases import PDF, APL, row
from local_storage import file_lock
from rag_observability import KEYS, Trace, CURRENT, measure

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(web, 'ROOT', tmp_path)
    monkeypatch.setattr(web, 'HISTORY_DIR', tmp_path)
    monkeypatch.setattr(web, '_session_documents', lambda sid: [{'doc_id': 'PDF_test'}])
    monkeypatch.setattr(web, 'is_administrative', lambda *a: 'administrative')
    monkeypatch.setattr(rag, 'check_model', lambda *a: 'mock')
    monkeypatch.setattr(rag, 'USE_REVIEW', False)
    monkeypatch.setattr(rag, 'FAST_PATH', False)
    web.app.config['TESTING'] = True
    return web.app.test_client()

@pytest.mark.parametrize('question', ['explique moi le pdf', 'explique-moi le document',
    'explique ce pdf', 'résume mon document', 'résumé du pdf', 'de quoi parle ce pdf',
    'que contient mon document', "c’est quoi ce document", 'analyse mon pdf',
    '  EXPLIQUE—MOI   le PDF ', 'résume mon pdf', 'de quoi parle ce document'])
def test_overview_bypasses_search_and_model(client, monkeypatch, question):
    monkeypatch.setattr(web.pdf_index, 'get_document_overview', lambda *a, **k: [row(PDF)])
    monkeypatch.setattr(web.pdf_index, 'search', lambda *a, **k: pytest.fail('search threshold must be bypassed'))
    monkeypatch.setattr(rag, 'request_json', lambda *a, **k: pytest.fail('no LLM for overview'))
    response = client.post('/api/chat', json={'question': question}).get_json()
    assert response['result']['answer_mode'] == 'document_overview'

def choose_first(path, payload, **kwargs):
    assert payload['think'] is False
    assert payload['options']['num_ctx'] == rag.NUM_CTX
    sources = json.loads(payload['messages'][-1]['content'])['sources']
    assert sum(len(s['evidence_units']) for s in sources) <= 6
    eid = sources[0]['evidence_units'][0]['evidence_id']
    return {'message': {'content': '<think>ignored reasoning</think>' + json.dumps({
        'status': 'answered', 'claims': [{'text': 'Le loyer est payé à la signature du bailleur.', 'evidence_ids': [eid]}]})}}

@pytest.mark.parametrize('question, expected, excluded', [
    ('Quel est mon loyer ?', 'Le loyer mensuel est de 450 euros, charges comprises.', 'signature'),
    ('Quel est le montant du dépôt de garantie ?', 'Le dépôt de garantie de 450 euros est versé à la signature.', 'loyer'),
    ('Quelle est la date limite de dépôt ?', '15 décembre 2026', 'signature'),
])
def test_display_is_exact(monkeypatch, question, expected, excluded):
    monkeypatch.setattr(rag, 'request_json', choose_first)
    monkeypatch.setattr(rag, 'USE_REVIEW', False)
    result = rag.generate(question, [row(PDF)])
    assert result['status'] == 'answered'
    text = ' '.join(c['text'] for c in result['claims'])
    assert expected in text and excluded not in text
    assert 'payé à la signature' not in rag.render(result)
    for claim in result['claims']:
        assert rag.normalize(claim['text']) in rag.normalize(PDF)

@pytest.mark.parametrize('question, first', [('Comment demander les APL ?', 'Pour demander'), ('Quel est le délai de l’APL ?', 'Le délai')])
def test_procedure_vs_deadline(question, first):
    ranked = rag.rank_evidence(question, rag.prepare_sources([row(APL, 'SP_apl')]))
    assert ranked[0]['evidence_units'][0]['quote'].startswith(first)
    if question.startswith('Comment'):
        assert all('délai' not in u['quote'] for s in ranked for u in s['evidence_units'])

def test_forged_quote_rejected():
    sources = rag.prepare_sources([row(PDF)])
    answer = {'status': 'answered', 'claims': [{'text': 'fake', 'evidence': [{'source_id': 'S1', 'quote': 'Le loyer est payé à la signature.'}]}]}
    with pytest.raises(ValueError, match='absente'):
        rag.validate_answer(answer, sources)

def test_no_legacy_mode_messages():
    for filename in ('app.py', 'answer_rag.py', 'internet_search.py'):
        tree = ast.parse(Path(filename).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value.split()) > 2:
                # Console comments/logs are not UI; ban all legacy instructional phrases.
                assert not re.search(r'(?i)(choisissez|essayez|source[^.]*).*\bmode\b', node.value)

def test_timings_and_private_log(client, tmp_path):
    data = client.post('/api/chat', json={'question': 'bonjour'}).get_json()
    timings = data['result']['timings']
    assert set(timings) == set(KEYS) | {'total'}
    assert sum(timings[k] for k in KEYS) <= timings['total']
    record = json.loads((tmp_path / 'data/timings.jsonl').read_text(encoding='utf-8'))
    assert 'question' not in record and 'bonjour' not in json.dumps(record)

def test_nested_timings_do_not_double_count():
    trace = Trace(); token = CURRENT.set(trace)
    try:
        with measure('base_search'):
            with measure('reranking'): time.sleep(.002)
    finally: CURRENT.reset(token)
    times = trace.result()
    assert sum(times[k] for k in KEYS) <= times['total']

def test_stream_order_and_lock(client, monkeypatch, tmp_path):
    def core():
        for key in ('pdf_search', 'base_search', 'reranking', 'llm_generation', 'review'):
            with measure(key): pass
        return web.jsonify({'answer': 'ok', 'result': {}})
    monkeypatch.setattr(web, '_chat_core', core)
    response = client.post('/api/chat/stream', json={'question': 'test'}, buffered=False)
    iterator = iter(response.response)
    first = json.loads(next(iterator))
    assert first['stage'] == 'pdf_search'
    with pytest.raises(TimeoutError):
        with file_lock(tmp_path / 'data/operation.lock', timeout=0): pass
    events = [first] + [json.loads(line) for line in iterator if line.strip()]
    assert [e.get('stage') for e in events[:-1]] == ['pdf_search', 'base_search', 'reranking', 'llm_generation', 'review']
    assert events[-1]['type'] == 'final'
    response.close()
    with file_lock(tmp_path / 'data/operation.lock', timeout=2): pass

def test_stream_error_releases_lock(client, monkeypatch, tmp_path):
    def fail(): raise RuntimeError('mock failure')
    monkeypatch.setattr(web, '_chat_core', fail)
    response = client.post('/api/chat/stream', json={'question': 'test'})
    assert json.loads(response.data)['type'] == 'error'
    with file_lock(tmp_path / 'data/operation.lock', timeout=2): pass

def test_disconnect_releases_lock_after_inflight_work(client, monkeypatch, tmp_path):
    started, finish = threading.Event(), threading.Event()
    def core():
        with measure('pdf_search'):
            started.set(); finish.wait(2)
        with measure('review'): pass
        return web.jsonify({'result': {}})
    monkeypatch.setattr(web, '_chat_core', core)
    response = client.post('/api/chat/stream', json={'question': 'test'}, buffered=False)
    assert started.wait(1)
    response.close(); finish.set()
    with file_lock(tmp_path / 'data/operation.lock', timeout=3): pass

def test_flags_default():
    tree = ast.parse(Path('answer_rag.py').read_text(encoding='utf-8'))
    assert "os.environ.get('RAG_FAST_PATH', '0')" in ast.unparse(tree)
    assert rag.PARAPHRASE is False

def test_fast_path_requires_calibrated_score(monkeypatch):
    sources = rag.rank_evidence('Quel est mon loyer ?', rag.prepare_sources([row(PDF)]))
    monkeypatch.setattr(rag, 'FAST_PATH', True)
    monkeypatch.delenv('RAG_FAST_MIN_SCORE', raising=False)
    assert rag.fast_answer(sources, 'Quel est mon loyer ?') is None
    monkeypatch.setenv('RAG_FAST_MIN_SCORE', '10')  # synthetic test values, not a recommendation
    assert rag.fast_answer(sources, 'Quel est mon loyer ?') is None
    sources[0]['rerank_score'] = 11
    result = rag.fast_answer(sources, 'Quel est mon loyer ?')
    assert result['fast_path_used'] and result['claims'][0]['text'] in PDF
    assert rag.fast_answer(sources, 'capitale du Japon') is None

def test_existing_ollama_is_not_restarted(monkeypatch):
    import start_ollama
    monkeypatch.setattr(start_ollama, 'server_version', lambda: '0.40.11')
    monkeypatch.setattr(start_ollama, 'warn_client_difference', lambda v: None)
    monkeypatch.setattr(start_ollama.subprocess, 'Popen', lambda *a, **k: pytest.fail('no server launch'))
    start_ollama.main()

def test_cache_and_configured_prewarm(monkeypatch):
    calls = []
    monkeypatch.setattr(web, 'retrieve_service_public_result', lambda q: calls.append(q) or {'reranked': []})
    web._retrieve_details_cached.cache_clear()
    web.retrieve_sp_cached('même question'); web.retrieve_sp_cached('même   question')
    assert len(calls) == 1
    monkeypatch.setattr(web, 'retrieve_service_public', lambda q: calls.append('E5+reranker'))
    monkeypatch.setattr(rag, 'MODEL', 'configured:test')
    monkeypatch.setattr(rag, 'request_json', lambda path, payload, **kw: calls.append(payload))
    web._prewarm()
    assert calls[-2] == 'E5+reranker'
    assert calls[-1]['model'] == 'configured:test'
    assert calls[-1]['options']['num_ctx'] == rag.NUM_CTX

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get('RAG_RUN_SLOW') != '1', reason='Set RAG_RUN_SLOW=1 for local Ollama')
def test_real_ollama_quotes():
    result = rag.generate('Quel est mon loyer ?', [row(PDF)])
    assert result['status'] == 'answered'
    assert all(rag.normalize(c['text']) in rag.normalize(PDF) for c in result['claims'])
