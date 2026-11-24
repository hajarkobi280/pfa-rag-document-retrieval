"""Reuse the responding system server; never install or update Ollama."""
import os
from pathlib import Path
import shutil
import subprocess
import time
import answer_rag

ROOT = Path(__file__).resolve().parent

def server_version():
    try:
        return answer_rag.request_json('/api/version', timeout=2).get('version')
    except RuntimeError:
        return None

def executable():
    candidates = [shutil.which('ollama'),
                  str(Path.home() / 'AppData/Local/Programs/Ollama/ollama.exe')]
    runtime = ROOT / '.runtime/ollama/ollama.exe'
    for value in candidates:
        if value and Path(value).is_file() and Path(value).resolve() != runtime.resolve():
            return Path(value)
    if runtime.is_file():
        print('Avertissement : utilisation du client de secours .runtime/ollama.', flush=True)
        return runtime
    raise RuntimeError('Ollama introuvable dans le PATH ou dans son installation Windows.')

def warn_client_difference(version):
    local = ROOT / '.runtime/ollama/ollama.exe'
    if local.exists():
        result = subprocess.run([str(local), '--version'], capture_output=True, text=True,
                                timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        import re
        versions = re.findall(r'\d+\.\d+\.\d+', result.stdout + result.stderr)
        if any(v != version for v in versions):
            print(f'Avertissement : serveur Ollama {version} ; client du projet : '
                  f'{", ".join(dict.fromkeys(versions))}. Le serveur système est conservé.', flush=True)

def main():
    version = server_version()
    if version:
        print(f'Serveur Ollama déjà disponible : {version}. Aucun serveur lancé.')
        warn_client_difference(version)
        return
    # Also avoid launching over a listening non-Ollama process.
    import socket
    try:
        with socket.create_connection(('127.0.0.1', 11434), timeout=1):
            print('Le port 11434 répond déjà. Aucun serveur lancé.')
            return
    except OSError:
        pass
    env = os.environ.copy()
    env.update(OLLAMA_KEEP_ALIVE='30m', OLLAMA_MAX_LOADED_MODELS='1',
               OLLAMA_NUM_PARALLEL='1', OLLAMA_FLASH_ATTENTION='1')
    # KV cache stays at its default until a separate measured comparison justifies q8_0.
    log = ROOT / 'data/ollama-server.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as output:
        subprocess.Popen([str(executable()), 'serve'], env=env, stdout=output,
                         stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    for _ in range(30):
        version = server_version()
        if version:
            print(f'Ollama disponible : {version}')
            warn_client_difference(version)
            return
        time.sleep(1)
    raise RuntimeError(f'Ollama ne répond pas ; consultez {log}.')

if __name__ == '__main__':
    main()
