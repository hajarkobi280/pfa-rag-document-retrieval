"""Recherche et lecture web avec nettoyage de requêtes et protections SSRF."""
from __future__ import annotations

from contextvars import copy_context
from rag_observability import stage, measure
import http.client
import html
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import socket
import ssl
import threading
import time
from urllib.error import URLError
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parent
# Dépendances web installées localement pour le runtime Python du projet.
_LOCAL_PACKAGE_PATH = ROOT / ".runtime" / "python-packages"
if _LOCAL_PACKAGE_PATH.is_dir():
    import sys
    sys.path.insert(0, str(_LOCAL_PACKAGE_PATH))
CACHE_PATH = ROOT / "data" / "internet_cache.json"
LOG_PATH = ROOT / "data" / "internet_queries.jsonl"
CACHE_TTL = 3600
MAX_PAGE_BYTES = 1024 * 1024
_cache_lock = threading.Lock()


class InternetSearchError(RuntimeError):
    pass


def sanitize_query(query: str) -> str:
    """Retire les identifiants fréquents avant toute transmission externe."""
    value = str(query or "")
    value = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", " ", value)
    value = re.sub(r"(?<!\w)\+?\d[\d\s()./-]{5,}\d(?!\w)", " ", value)
    value = re.sub(r"\d{6,}", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_query(query: str) -> str:
    return " ".join(sanitize_query(query).casefold().split())


def _cache_read():
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        now = time.time()
        return {k: v for k, v in payload.items() if v.get("expires", 0) > now}
    except (OSError, ValueError, TypeError):
        return {}


def _cache_write(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _log_query(query):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "query": query}, ensure_ascii=False)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _provider_search(query, max_results=6):
    searx = os.environ.get("RAG_SEARXNG_URL", "").strip()
    brave_key = os.environ.get("RAG_BRAVE_API_KEY", "").strip()
    tavily_key = os.environ.get("RAG_TAVILY_API_KEY", "").strip()
    if searx:
        _log_query(query)
        url = searx.rstrip("/") + "/search?" + urlencode({"q": query, "format": "json"})
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "LocalAdminRAG/1.0"})
        with build_opener(ProxyHandler({})).open(req, timeout=8) as response:
            data = json.loads(response.read(1_000_000).decode("utf-8"))
        return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")} for x in data.get("results", [])[:max_results]]
    if brave_key:
        _log_query(query)
        url = "https://api.search.brave.com/res/v1/web/search?" + urlencode({"q": query, "count": max_results})
        req = Request(url, headers={"Accept": "application/json", "X-Subscription-Token": brave_key, "User-Agent": "LocalAdminRAG/1.0"})
        with build_opener(ProxyHandler({})).open(req, timeout=8) as response:
            data = json.loads(response.read(1_000_000).decode("utf-8"))
        return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")} for x in data.get("web", {}).get("results", [])[:max_results]]
    if tavily_key:
        _log_query(query)
        req = Request("https://api.tavily.com/search", data=json.dumps({"api_key": tavily_key, "query": query, "max_results": max_results, "search_depth": "basic"}).encode(), headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        with build_opener(ProxyHandler({})).open(req, timeout=8) as response:
            data = json.loads(response.read(1_000_000).decode("utf-8"))
        return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")} for x in data.get("results", [])[:max_results]]
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise InternetSearchError("Le fournisseur DDGS n'est pas installé. Installez `ddgs` (Python 3.14 compatible), ou configurez RAG_SEARXNG_URL, RAG_BRAVE_API_KEY ou RAG_TAVILY_API_KEY.") from exc
    try:
        _log_query(query)
        found = DDGS(timeout=8).text(query, max_results=max_results)
    except Exception as exc:
        raise InternetSearchError(f"Le moteur DDGS est indisponible : {exc}") from exc
    return [{"title": x.get("title", ""), "url": x.get("href", x.get("url", "")), "snippet": x.get("body", x.get("snippet", ""))} for x in found]


def _official(url: str) -> bool:
    try:
        configured = json.loads((ROOT / "official_domains.json").read_text(encoding="utf-8")).get("domains", [])
    except (OSError, ValueError):
        configured = []
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in configured)


@stage('internet_search')
def search(query, max_results=6):
    """Retourne {title,url,snippet,official}; query nettoyée uniquement."""
    clean = sanitize_query(query)
    if not clean:
        return []
    key = normalize_query(clean)
    with _cache_lock:
        cache = _cache_read()
        if key in cache:
            return [{**row, "external_query_sent": False} for row in cache[key]["results"][:max_results]]
    results = _provider_search(clean, max_results=max_results)
    safe = []
    for row in results:
        url = str(row.get("url", "")).strip()
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            continue
        safe.append({"title": str(row.get("title", ""))[:300], "url": url,
                     "snippet": str(row.get("snippet", ""))[:1200], "official": _official(url),
                     "external_query_sent": True})
    safe.sort(key=lambda r: (not r["official"],))
    with _cache_lock:
        cache[key] = {"expires": time.time() + CACHE_TTL,
                      "results": [{k:v for k,v in row.items() if k != "external_query_sent"} for row in safe]}
        _cache_write(cache)
    return safe[:max_results]


def _public_addresses(host, port):
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise InternetSearchError(f"Impossible de résoudre le site {host}.") from exc
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses:
        raise InternetSearchError(f"Aucune adresse publique trouvée pour {host}.")
    for value in addresses:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
        if not ip.is_global:
            raise InternetSearchError("Accès refusé : l'adresse du site est privée ou réservée.")
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, ip, port, timeout):
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())
        self._resolved_ip = ip

    def connect(self):
        raw = socket.create_connection((self._resolved_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


class _TextExtractor(HTMLParser):
    DROP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "button"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title_parts, self.stack = [], [], []
        self.in_title = False
    def handle_starttag(self, tag, attrs):
        if tag in self.DROP:
            self.stack.append(tag)
        if tag == "title": self.in_title = True
        if tag in {"p", "br", "li", "h1", "h2", "h3", "section", "article"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag == "title": self.in_title = False
        if tag in self.stack:
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[index:]
        if tag in {"p", "br", "li", "h1", "h2", "h3", "section", "article"}: self.parts.append("\n")
    def handle_data(self, data):
        if self.stack: return
        value = html.unescape(data).strip()
        if value:
            self.parts.append(value)
            if self.in_title: self.title_parts.append(value)


def fetch_page(url, timeout=8, max_bytes=MAX_PAGE_BYTES, max_redirects=3):
    """Télécharge une page publique en épinglant chaque DNS avant connexion."""
    current = url
    for redirect in range(max_redirects + 1):
        parsed = urlsplit(current)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise InternetSearchError("URL de page non autorisée.")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in (80, 443):
            raise InternetSearchError("Port non autorisé pour le téléchargement.")
        addresses = _public_addresses(parsed.hostname, port)
        conn = None
        try:
            if parsed.scheme == "https":
                conn = _PinnedHTTPSConnection(parsed.hostname, addresses[0], port, min(8, timeout))
            else:
                conn = http.client.HTTPConnection(parsed.hostname, port=port, timeout=min(8, timeout))
                original_connect = conn.connect
                def pinned_connect():
                    conn.sock = socket.create_connection((addresses[0], port), conn.timeout)
                conn.connect = pinned_connect
            path = urlunsplit(("", "", quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~"),
                               quote(parsed.query, safe="%=&?/:@!$'()*+,;~-._"), ""))
            conn.request("GET", path, headers={"Host": parsed.hostname, "User-Agent": "LocalAdminRAG/1.0", "Accept": "text/html,text/plain;q=0.9", "Accept-Encoding": "identity", "Connection": "close"})
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                response.read(1024)
                if not location or redirect >= max_redirects:
                    raise InternetSearchError("Trop de redirections ou redirection invalide.")
                current = urljoin(current, location)
                continue
            if response.status < 200 or response.status >= 300:
                raise InternetSearchError(f"Le site a répondu HTTP {response.status}.")
            content_type = response.getheader("Content-Type", "").lower()
            if content_type and not any(x in content_type for x in ("text/html", "text/plain", "application/xhtml")):
                raise InternetSearchError("Le contenu du site n'est pas une page texte HTML.")
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                data = data[:max_bytes]
            charset = "utf-8"
            match = re.search(r"charset=([\w-]+)", content_type)
            if match: charset = match.group(1)
            source = data.decode(charset, errors="replace")
            if "html" not in content_type and not source.lstrip().startswith("<"):
                return {"title": parsed.hostname, "url": current, "text": re.sub(r"\s+", " ", source).strip()}
            parser = _TextExtractor()
            parser.feed(source)
            title = " ".join(parser.title_parts).strip() or parsed.hostname
            raw_lines = [re.sub(r"\s+", " ", line).strip() for line in "\n".join(parser.parts).splitlines()]
            lines = [line for line in raw_lines if line and not (len(line) <= 90 and not re.search(r"[.!?;:]", line))]
            text = "\n".join(lines)
            try:
                import trafilatura
                extracted = trafilatura.extract(source, include_comments=False, include_tables=True)
                if extracted and len(extracted) > len(text) * 0.5:
                    text = extracted
            except ImportError:
                pass
            return {"title": title[:300], "url": current, "text": text[:MAX_PAGE_BYTES]}
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise InternetSearchError(f"Impossible de télécharger la page : {exc}") from exc
        finally:
            if conn is not None:
                conn.close()
    raise InternetSearchError("Trop de redirections.")


def search_pages(query, max_results=6, page_limit=3, timeout=20):
    """Essaie les résultats en parallèle et conserve au plus page_limit pages."""
    started = time.monotonic()
    page_key = "pages:" + normalize_query(query)
    with _cache_lock:
        cached_pages = _cache_read().get(page_key)
        if cached_pages and cached_pages.get("expires", 0) > time.time():
            return [{**page, "external_query_sent": False} for page in cached_pages.get("pages", [])[:page_limit]]
    results = run_with_timeout(lambda: search(query, max_results=max_results),
                               timeout=min(10, max(0.01, timeout / 2)))
    if not results:
        return []
    pages = []
    errors = []
    completed = queue.Queue()
    deadline = started + timeout
    def download(index, result):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InternetSearchError("Délai de lecture des pages dépassé.")
            page = fetch_page(result["url"], timeout=min(8, remaining))
            page.update(official=_official(page["url"]), snippet=result.get("snippet", ""))
            completed.put((index, page, None))
        except Exception as exc:
            completed.put((index, None, str(exc)))
    with measure('page_download'):
        for index, result in enumerate(results):
            threading.Thread(target=download, args=(index, result), daemon=True).start()
        downloaded = []
        for _ in results:
            remaining = deadline - time.monotonic() - 0.1
            if remaining <= 0: break
            try:
                index, page, error = completed.get(timeout=remaining)
            except queue.Empty:
                break
            if error: errors.append(error)
            if page and page.get("text", "").strip(): downloaded.append((index, page))
        pages = [page for _, page in sorted(downloaded)][:page_limit]
    if not pages:
        raise InternetSearchError("Des résultats ont été trouvés, mais les pages sont inaccessibles ou sans texte exploitable. Réessayez avec une question plus précise.")
    query_sent = any(row.get("external_query_sent", False) for row in results)
    for page in pages:
        page["external_query_sent"] = query_sent
    with _cache_lock:
        cache = _cache_read()
        cache[page_key] = {"expires": time.time() + CACHE_TTL,
                           "pages": [{k:v for k,v in page.items() if k != "external_query_sent"} for page in pages]}
        _cache_write(cache)
    return pages


def run_with_timeout(function, timeout=20):
    """Borne l'attente HTTP sans bloquer le thread Flask lors d'une panne."""
    result = queue.Queue(maxsize=1)
    context = copy_context()
    def worker():
        try: result.put((True, context.run(function)))
        except BaseException as exc: result.put((False, exc))
    thread = threading.Thread(target=worker, name="internet-search", daemon=True)
    thread.start()
    thread.join(timeout=max(0.01, timeout))
    if thread.is_alive():
        raise InternetSearchError(f"La recherche Internet a dépassé {timeout:g} secondes. Réessayez plus tard.")
    ok, value = result.get_nowait()
    if ok: return value
    raise value
