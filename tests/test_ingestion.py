from pathlib import Path
from app.ingestion import load_corpus

def test_complete_corpus_is_loaded():
    items=load_corpus(Path('data'))
    assert len(items)==40
    assert sum(x.source_type=='faq' for x in items)==15
    assert sum(x.source_type=='policy' for x in items)==10
    assert sum(x.source_type=='ticket' for x in items)==15

def test_current_refund_policy_is_present():
    policy=next(x for x in load_corpus(Path('data')) if x.document_id=='POLICY-02')
    assert '14 days' in policy.content
    assert '7-day' in policy.content
    assert policy.contains_deprecation_notice is True
