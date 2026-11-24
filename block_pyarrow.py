"""
Shim anti-crash : pyarrow (arrow.dll) n'est pas compatible avec Python 3.14.
Ce script doit être importé AVANT tout autre import (première ligne après sys).

Stratégie :
1. Bloquer pyarrow et tous ses sous-modules → ImportError propre
2. Bloquer le chargement du vrai paquet 'datasets' (HuggingFace) → stub léger
3. Patcher transformers pour désactiver is_datasets_available()
   → transformers.Trainer ne sera jamais importé dans ce chemin
"""
import sys
import importlib.util
from types import ModuleType


# ---------------------------------------------------------------------------
# 1. Bloquer pyarrow (arrow.dll crashe sur Python 3.14)
# ---------------------------------------------------------------------------
class _BlockArrow:
    """Meta path finder qui bloque pyarrow sans le charger."""
    _BLOCKED = frozenset({"pyarrow"})

    def find_spec(self, fullname, path, target=None):
        if fullname in self._BLOCKED or fullname.startswith("pyarrow."):
            raise ImportError(
                f"[pyarrow-block] {fullname} est désactivé sur cette machine "
                "(arrow.dll incompatible avec Python 3.14)."
            )
        return None


if not any(isinstance(f, _BlockArrow) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockArrow())

# Purger toute version déjà chargée
for _mod in list(sys.modules):
    if _mod == "pyarrow" or _mod.startswith("pyarrow."):
        del sys.modules[_mod]


# ---------------------------------------------------------------------------
# 2. Stub complet pour 'datasets' (HuggingFace) + tous ses sous-modules
#    On remplace le VRAI paquet installé par un stub inerte.
# ---------------------------------------------------------------------------
def _make_stub(name: str) -> ModuleType:
    m = ModuleType(name)
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    m.__version__ = "0.0.0"
    m.__path__ = []          # marque comme package
    m.__package__ = name
    m.__loader__ = None
    return m


class _BlockDatasets:
    """Intercepte l'import du vrai paquet datasets et retourne un stub."""
    _PREFIXES = ("datasets",)

    def find_spec(self, fullname, path, target=None):
        for p in self._PREFIXES:
            if fullname == p or fullname.startswith(p + "."):
                # Retourner un ModuleSpec qui pointe sur notre stub loader
                return importlib.util.spec_from_loader(fullname, loader=_StubLoader())
        return None


class _StubLoader:
    def create_module(self, spec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        return _make_stub(spec.name)

    def exec_module(self, module):
        sys.modules[module.__name__] = module


if not any(isinstance(f, _BlockDatasets) for f in sys.meta_path):
    sys.meta_path.insert(1, _BlockDatasets())  # après _BlockArrow

# Purger le vrai paquet datasets s'il est déjà partiellement chargé
for _mod in list(sys.modules):
    if _mod == "datasets" or _mod.startswith("datasets."):
        del sys.modules[_mod]

# Pré-remplir le stub principal
sys.modules["datasets"] = _make_stub("datasets")


# ---------------------------------------------------------------------------
# 3. Patch transformers.utils.import_utils APRÈS que transformers soit chargé
#    → is_datasets_available() retournera False proprement
# ---------------------------------------------------------------------------
def _patch_transformers_if_loaded():
    """Appelé après le premier import de transformers pour patcher la fonction."""
    try:
        import transformers.utils.import_utils as _iu
        _iu._datasets_available = False
        if hasattr(_iu, "is_datasets_available"):
            _iu.is_datasets_available = lambda: False
    except Exception:
        pass


# Installer un hook post-import pour transformers
class _TransformersPatcher:
    """Meta path finder qui patche transformers une fois chargé."""
    _done = False

    def find_spec(self, fullname, path, target=None):
        return None

    def find_module(self, fullname, path=None):
        return None

    @staticmethod
    def exec_module_hook():
        if not _TransformersPatcher._done:
            _TransformersPatcher._done = True
            _patch_transformers_if_loaded()


# On patche aussi directement si transformers est déjà importé
_patch_transformers_if_loaded()
