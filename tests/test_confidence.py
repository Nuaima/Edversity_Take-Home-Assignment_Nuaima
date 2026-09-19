from app.confidence import ConfidenceEngine
from app.schemas import RetrievedChunk

def result(document_id='POLICY-02',source_type='policy',similarity=0.75,status=None):
    return RetrievedChunk(chunk_id=document_id.lower()+'-001',document_id=document_id,source_type=source_type,title='Example',content='Current policy evidence.',updated_at='January 2026',status=status,authority=1.0 if source_type=='policy' else 0.72,freshness=1.0,contains_deprecation_notice=False,similarity=similarity,rerank_score=similarity)

def test_ambiguous_cancel_is_not_guessed():
    d=ConfidenceEngine().assess('Cancel my LearnForge.',[result()])
    assert d.escalate and d.clarification

def test_promotional_exception_escalates():
    assert ConfidenceEngine().assess('I bought it 21 days ago but had a 30-day guarantee.',[result()]).escalate

def test_strong_current_policy_can_answer():
    d=ConfidenceEngine().assess('What is the standard course refund window?',[result(),result('FAQ-02','faq',0.7)])
    assert d.escalate is False
