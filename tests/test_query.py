from app.query import build_retrieval_query
from app.schemas import ChatTurn

def test_short_followup_uses_previous_user_turn():
    history=[ChatTurn(role='user',content='Can I get a refund for an individual course?'),ChatTurn(role='assistant',content='The standard window is generally 14 days.')]
    query=build_retrieval_query('What if I bought mine 20 days ago?',history)
    assert 'Can I get a refund' in query and '20 days' in query

def test_standalone_question_stays_unchanged():
    message='Which modern browsers are supported by LearnForge and is Internet Explorer supported?'
    assert build_retrieval_query(message,[])==message
