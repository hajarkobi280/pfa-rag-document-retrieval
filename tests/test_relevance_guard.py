import pytest
import answer_rag as rag
import app
from benchmark_cases import CASES, PDF, row
from calibrate_topic_guard import TRAIN, HOLDOUT, evaluate
from relevance_guard import relevant

@pytest.mark.parametrize('case', [c for c in CASES if c.get('abstain')])
def test_irrelevant_evidence_abstains_without_llm(monkeypatch, case):
    monkeypatch.setattr(rag, 'request_json', lambda *a, **k: pytest.fail('No LLM for unrelated evidence'))
    result = rag.generate(case['question'], case['rows'])
    assert result['status'] == 'insufficient'
    assert result['validation'] == 'no_evidence_units'

@pytest.mark.parametrize('question,quote,expected', [c for c in HOLDOUT if c[2]])
def test_holdout_positive_topics_remain_available(question, quote, expected):
    assert relevant(question, quote)

@pytest.mark.parametrize('question,quote,expected', [c for c in HOLDOUT if not c[2] and 'mariage' not in c[0]])
def test_holdout_unrelated_topics_rejected(question, quote, expected):
    assert not relevant(question, quote)

def test_calibration_exposes_remaining_false_acceptance():
    report = evaluate(HOLDOUT, 0)
    assert report['false_refusals'] == 0
    assert report['false_acceptances'] == 1  # known limitation, not hidden by calibration

def test_title_cannot_validate_unrelated_quote():
    result = {'claims':[{'evidence':[{'source_id':'S1','quote':'Le loyer est de 450 euros.'}]}],
              'sources':[{'source_id':'S1','title':'Passeport','text':'Le loyer est de 450 euros.'}]}
    assert not app._cited_evidence_matches('Comment demander un passeport ?', result)

def test_no_answer_message():
    assert 'trouvé cette information' in rag.render({'status':'insufficient'})
