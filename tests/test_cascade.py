import json
import socket
import pytest
import app as app_module
import answer_rag
import internet_search

REAL_SEARCH_LOCAL = app_module._search_local

@pytest.fixture
def client(monkeypatch, tmp_path):
    app_module.app.config["TESTING"] = True
    monkeypatch.setattr(app_module, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(app_module, "ROOT", tmp_path)
    monkeypatch.setattr(app_module.pdf_index, "list_documents", lambda: [])
    # These tests isolate cascade decisions; session isolation is tested separately.
    monkeypatch.setattr(app_module, "_session_documents", lambda sid: app_module.pdf_index.list_documents())
    monkeypatch.setattr(app_module, "_search_local", lambda *a, **k: ([], 0.1, None))
    return app_module.app.test_client()

def ask(client, question, mode="auto", doc_id=None):
    return client.post("/api/chat", json={"question":question,"mode":mode,"doc_id":doc_id}).get_json()

def answered(rows, mode="extractive"):
    sources=[]
    for i,row in enumerate(rows,1):
        sources.append({"source_id":f"S{i}","chunk_id":row["chunk_id"],"title":row.get("title","Fiche officielle"),"url":row.get("url","https://service-public.gouv.fr/test"),"text":row.get("text",row.get("chunk_text","preuve pertinente")),"official":row.get("official",False)})
    return {"status":"answered","claims":[{"text":"Déposez une demande selon les étapes indiquées.","evidence":[{"source_id":"S1","quote":rows[0].get("text",rows[0].get("chunk_text","preuve pertinente"))}]}],"sources":sources,"answer_mode":mode}

def mock_internet(monkeypatch, pages=None):
    calls=[]
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda fn,timeout=20: fn())
    def search_pages(query,**kwargs):
        calls.append(query)
        return pages or [{"title":"CNSS — informations officielles","url":"https://www.cnss.ma/","text":"La CNSS assure la protection sociale et présente les services accessibles aux affiliés.","official":True,"external_query_sent":True}]
    monkeypatch.setattr(app_module.internet_search,"search_pages",search_pages)
    monkeypatch.setattr(app_module,"_rank_web_passages",lambda pages,q:[{"chunk_id":"WEB_1_0","title":pages[0]["title"],"url":pages[0]["url"],"text":pages[0]["text"],"scored_excerpt":pages[0]["text"],"official":pages[0].get("official",False)}])
    monkeypatch.setattr(answer_rag,"check_model",lambda model:"mock-digest")
    monkeypatch.setattr(answer_rag,"generate_web",lambda q,rows,model:answered(rows,"web"))
    return calls


def test_auto_uses_service_public_before_internet(client,monkeypatch):
    row={"chunk_id":"SP_1","title":"Aide personnalisée au logement (APL)","url":"https://service-public.gouv.fr/apl","text":"Pour demander les APL, déposez votre dossier auprès de la Caf."}
    monkeypatch.setattr(app_module,"_search_local",lambda *a,**k:([row],0.86,None))
    monkeypatch.setattr(answer_rag,"check_model",lambda model:"digest")
    monkeypatch.setattr(answer_rag,"generate",lambda q,rows,model:answered(rows))
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda *a,**k:pytest.fail("Internet ne doit pas être appelé"))
    data=ask(client,"Comment demander les APL ?")
    assert data["result"]["source_used"]=="service_public"
    assert data["result"]["status"]=="answered"


def test_auto_uses_internet_for_cnss_outside_base(client,monkeypatch):
    calls=mock_internet(monkeypatch)
    data=ask(client,"c est quoi la cnss")
    assert data["result"]["status"]=="answered"
    assert data["result"]["source_used"]=="internet"
    assert "Maroc" in calls[0]
    assert data["result"]["sources"][0]["url"]=="https://www.cnss.ma/"


def test_auto_pdf_only_question_uses_pdf(client,monkeypatch):
    monkeypatch.setattr(app_module.pdf_index,"list_documents",lambda:[{"doc_id":"PDF_test"}])
    row={"chunk_id":"PDF_test_1","title":"Dossier aide logement","url":"local.pdf","text":"Le loyer mensuel indiqué dans le dossier est de 450 euros."}
    monkeypatch.setattr(app_module,"_search_local",lambda *a,**k:([row],0.3,None))
    monkeypatch.setattr(answer_rag,"check_model",lambda model:"digest")
    monkeypatch.setattr(answer_rag,"generate",lambda q,rows,model:answered(rows))
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda *a,**k:pytest.fail("Internet ne doit pas être appelé"))
    data=ask(client,"Quel est le montant du loyer ?")
    assert data["result"]["source_used"]=="pdf"


def test_hybrid_filters_irrelevant_pdf_and_uses_service_public(client,monkeypatch):
    monkeypatch.setattr(app_module, "_search_local", REAL_SEARCH_LOCAL)
    docs=[{"doc_id":"PDF_test"}]
    monkeypatch.setattr(app_module.pdf_index,"list_documents",lambda:docs)
    def pdf_search(*a,**kw):
        assert kw["min_score"]==0.8
        return []
    monkeypatch.setenv("RAG_PDF_MIN_HYBRID","0.8")
    monkeypatch.setattr(app_module.pdf_index,"search",pdf_search)
    sp={"chunk_id":"SP_1","title":"Aide personnalisée au logement (APL)","url":"https://service-public.gouv.fr/apl","text":"Pour demander les APL, déposez un dossier auprès de la Caf."}
    monkeypatch.setattr(app_module,"retrieve_service_public_result",lambda q:{"reranked":[sp],"semantic_top_score":0.86})
    monkeypatch.setattr(answer_rag,"check_model",lambda model:"digest")
    monkeypatch.setattr(answer_rag,"generate",lambda q,rows,model:answered(rows))
    data=ask(client,"Comment demander les APL ?")
    assert data["result"]["source_used"]=="service_public"


def test_auto_with_pdf_and_missing_local_data_falls_back_to_internet(client,monkeypatch):
    monkeypatch.setattr(app_module.pdf_index,"list_documents",lambda:[{"doc_id":"PDF_test"}])
    mock_internet(monkeypatch)
    data=ask(client,"c est quoi la cnss")
    assert data["result"]["source_used"]=="internet"


@pytest.mark.parametrize("question",["recette de couscous","quelle est la capitale du Japon","écris-moi un poème sur la mer","qui a gagné la coupe du monde 2018"])
def test_out_of_domain_never_calls_generation_or_internet(client,monkeypatch,question):
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda *a,**k:pytest.fail("Internet ne doit pas être appelé"))
    monkeypatch.setattr(answer_rag,"generate",lambda *a,**k:pytest.fail("Génération ne doit pas être appelée"))
    data=ask(client,question)
    assert data["result"]["status"]=="out_of_domain"
    assert data["result"]["source_used"]=="none"


def test_extractive_abstention_falls_back_to_internet(client,monkeypatch):
    row={"chunk_id":"SP_1","title":"Démarche","url":"https://service-public.gouv.fr","text":"La CNSS est un organisme administratif. Texte suffisamment long pour générer une source utile."}
    monkeypatch.setattr(app_module,"_search_local",lambda *a,**k:([row],0.8,None))
    monkeypatch.setattr(answer_rag,"check_model",lambda model:"digest")
    monkeypatch.setattr(answer_rag,"generate",lambda *a,**k:{"status":"insufficient","claims":[],"sources":[]})
    mock_internet(monkeypatch)
    assert ask(client,"c est quoi la cnss")["result"]["source_used"]=="internet"



def test_auto_explicit_document_request_stays_pdf_and_never_uses_internet(client,monkeypatch):
    monkeypatch.setattr(app_module.pdf_index,"list_documents",lambda:[{"doc_id":"PDF_test"}])
    seen=[]
    monkeypatch.setattr(app_module,"_search_local",lambda q,strategy,*a,**k:seen.append(strategy) or ([],None,None))
    monkeypatch.setattr(answer_rag,"request_json",lambda *a,**k:{"message":{"content":"{\"domain\":\"administrative\"}"}})
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda *a,**k:pytest.fail("document explicit: no Internet"))
    data=ask(client,"Quel est le montant dans mon document ?")
    assert seen==["pdf"]
    assert data["result"]["source_used"]=="none"
    assert "information dans le document" in data["answer"]


def test_manual_pdf_no_match_suggests_rephrasing(client,monkeypatch):
    monkeypatch.setattr(app_module.pdf_index,"list_documents",lambda:[{"doc_id":"PDF_test"}])
    monkeypatch.setattr(app_module,"_search_local",lambda *a,**k:([],None,None))
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda *a,**k:pytest.fail("manual PDF must stay local"))
    data=ask(client,"Quel est le montant du loyer ?",mode="pdf")
    assert "information dans le document" in data["answer"]
    assert "mode" not in data["answer"].lower()


def test_manual_service_public_is_strict(client,monkeypatch):
    monkeypatch.setattr(app_module,"_search_local",lambda *a,**k:([],0.1,None))
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda *a,**k:pytest.fail("mode manuel ne doit pas replier"))
    data=ask(client,"c est quoi la cnss",mode="service_public")
    assert "mode" not in data["answer"].lower()
    assert data["result"]["source_used"]=="none"


def test_manual_internet_does_not_search_local(client,monkeypatch):
    calls=mock_internet(monkeypatch)
    monkeypatch.setattr(app_module,"_search_local",lambda *a,**k:pytest.fail("base locale non interrogée"))
    data=ask(client,"c est quoi la cnss",mode="internet")
    assert data["result"]["source_used"]=="internet"
    assert calls


@pytest.mark.parametrize("mode",["auto","internet"])
def test_input_accepts_new_modes(client,mode):
    result=client.post("/api/chat",json={"question":"merci","mode":mode})
    assert result.status_code==200

def test_input_rejects_unknown_mode(client):
    result=client.post("/api/chat",json={"question":"bonjour","mode":"moon"})
    assert result.status_code==400


def test_only_sanitized_question_is_sent_not_pdf_or_history(client,monkeypatch):
    captured=mock_internet(monkeypatch)
    query="c est quoi CNSS, contact prenom.nom@example.com au 06 12 34 56 78 dossier 123456789"
    data=ask(client,query)
    sent=captured[0]
    assert "prenom.nom@example.com" not in sent
    assert "06 12 34 56 78" not in sent
    assert "123456789" not in sent
    assert "PDF_PRIVATE_TEXT" not in sent
    assert "Maroc" in sent
    assert data["result"]["external_query_sent"]



def test_search_provider_receives_only_sanitized_text(monkeypatch,tmp_path):
    monkeypatch.setattr(internet_search,"CACHE_PATH",tmp_path/"cache.json")
    monkeypatch.setattr(internet_search,"LOG_PATH",tmp_path/"queries.jsonl")
    sent=[]
    def provider(query,max_results):
        sent.append(query); internet_search._log_query(query)
        return [{"title":"Official","url":"https://service-public.gouv.fr/a","snippet":"snippet"}]
    monkeypatch.setattr(internet_search,"_provider_search",provider)
    internet_search.search("demande CNSS a@b.example 06 12 34 56 78 dossier 123456789")
    assert "a@b.example" not in sent[0] and "06 12 34 56 78" not in sent[0] and "123456789" not in sent[0]
    log=(tmp_path/"queries.jsonl").read_text(encoding="utf-8")
    assert "a@b.example" not in log and "123456789" not in log


def test_ssrf_private_ip_rejected(monkeypatch):
    def private(*a,**k):
        return [(socket.AF_INET,socket.SOCK_STREAM,6,"",("127.0.0.1",443))]
    monkeypatch.setattr(internet_search.socket,"getaddrinfo",private)
    with pytest.raises(internet_search.InternetSearchError):
        internet_search.fetch_page("https://127.0.0.1/private")


def test_ssrf_redirect_to_private_rejected(monkeypatch):
    def resolve(host,port,*a,**k):
        ip="127.0.0.1" if host=="private.test" else "93.184.216.34"
        return [(socket.AF_INET,socket.SOCK_STREAM,6,"",(ip,port))]
    class Response:
        status=302
        def getheader(self,name,default=None): return "https://private.test/secret" if name=="Location" else default
        def read(self,n=-1): return b""
    class FakeConnection:
        def __init__(self,*a,**k): pass
        def request(self,*a,**k): pass
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(internet_search.socket,"getaddrinfo",resolve)
    monkeypatch.setattr(internet_search,"_PinnedHTTPSConnection",FakeConnection)
    with pytest.raises(internet_search.InternetSearchError):
        internet_search.fetch_page("https://public.test/start")


def test_internet_claim_with_invented_url_is_rejected(monkeypatch):
    source={"chunk_id":"WEB_1_0","title":"Page","url":"https://official.gov.fr/","text":"La démarche est présentée sur la page officielle avec ses étapes."}
    payload={"status":"answered","claims":[{"text":"Consultez https://invented.example","evidence_ids":["S1:E1"]}]}
    monkeypatch.setattr(answer_rag,"request_json",lambda *a,**k:{"message":{"content":json.dumps(payload)}})
    result=answer_rag.generate_web("Comment faire cette démarche ?",[source])
    assert result["status"]=="validation_failed"
    assert "invented.example" not in answer_rag.render(result)


def test_network_failure_is_a_clear_result(client,monkeypatch):
    monkeypatch.setattr(app_module.internet_search,"run_with_timeout",lambda *a,**k:(_ for _ in ()).throw(internet_search.InternetSearchError("timeout")))
    data=ask(client,"c est quoi la cnss",mode="internet")
    assert data["result"]["status"]=="internet_unavailable"
    assert "Internet est indisponible" in data["answer"]


def test_web_ranking_preserves_best_passage_from_third_page(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    class Encoder:
        def encode(self, texts, **kwargs):
            if texts[0].startswith("query:"):
                return np.array([[1.0]])
            return np.array([[1.0 - i * 0.01] for i in range(len(texts))])
    monkeypatch.setattr(app_module, "_searcher", SimpleNamespace(engine=SimpleNamespace(model=Encoder())))
    pages = [{"url": f"https://example.com/{i}", "text": f"PAGE{i} " + "texte pertinent. " * 120} for i in range(3)]
    rows = app_module._rank_web_passages(pages, "texte pertinent")
    assert len(rows) == 3
    assert all(row["text"].startswith(f"PAGE{i}") for i, row in enumerate(rows))
    assert sum(len(row["text"]) for row in rows) <= 3000


def test_web_generation_refusal_is_not_reported_as_network_failure(client, monkeypatch):
    mock_internet(monkeypatch)
    monkeypatch.setattr(answer_rag, "generate_web", lambda *a: {"status":"validation_failed", "validation":"Abstention avec affirmations"})
    data = ask(client, "c est quoi la cnss", mode="internet")
    assert "citations valides" in data["answer"]
    assert "Internet est indisponible" not in data["answer"]


def test_classifier_error_defaults_to_administrative(monkeypatch):
    monkeypatch.setattr(answer_rag,"request_json",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("offline")))
    assert app_module.is_administrative("qui s'occupe de ma situation particulière") == "administrative"


def test_heading_filter_skips_service_public_but_not_pdf():
    row={"title":"Titre de démarche","chunk_text":"Titre de démarche","url":"local","context":[]}
    assert answer_rag.prepare_sources([{**row,"chunk_id":"SP_1"}])==[]
    assert answer_rag.prepare_sources([{**row,"chunk_id":"PDF_1"}])

@pytest.mark.slow
@pytest.mark.skipif(__import__("os").environ.get("RAG_RUN_SLOW")!="1",reason="set RAG_RUN_SLOW=1 for local models and indexes")
def test_slow_real_search():
    result=app_module.retrieve_service_public_result("Comment demander les APL ?")
    assert result.get("semantic_top_score") is not None

