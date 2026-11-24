"""Request-local wall-clock measurements and progress; no question text by default."""
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import threading
import time

KEYS = ('classification', 'pdf_search', 'base_search', 'reranking',
        'llm_generation', 'review', 'internet_search', 'page_download')
LABELS = {'classification': 'Analyse de la question...', 'pdf_search': 'Recherche dans le PDF...',
          'base_search': 'Recherche dans la base...', 'reranking': 'Reclassement...',
          'llm_generation': 'Rédaction...', 'review': 'Vérification des citations...',
          'internet_search': 'Recherche sur Internet...', 'page_download': 'Lecture des pages...'}
CURRENT = ContextVar('rag_trace', default=None)
STACK = ContextVar('rag_trace_stack', default=())

class Cancelled(Exception):
    pass

class Trace:
    def __init__(self, emit=None, cancelled=None):
        self.started = time.perf_counter()
        self.timings = dict.fromkeys(KEYS, 0.0)
        self.emit = emit
        self.last_stage = None
        self.cancelled = cancelled or threading.Event()

    def progress(self, key):
        if self.cancelled.is_set():
            raise Cancelled('Client déconnecté')
        if self.emit and self.last_stage != key:
            self.emit({'type': 'stage', 'stage': key, 'message': LABELS[key]})
        self.last_stage = key

    def result(self):
        return {**self.timings, 'total': time.perf_counter() - self.started}

@contextmanager
def measure(key):
    trace = CURRENT.get()
    if trace is None:
        yield
        return
    trace.progress(key)
    parent = STACK.get()
    frame = [time.perf_counter(), 0.0]
    token = STACK.set((*parent, frame))
    try:
        yield
    finally:
        elapsed = time.perf_counter() - frame[0]
        trace.timings[key] += max(0.0, elapsed - frame[1])
        if parent:
            parent[-1][1] += elapsed
        STACK.reset(token)

def stage(key):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with measure(key):
                return function(*args, **kwargs)
        return wrapped
    return decorate
