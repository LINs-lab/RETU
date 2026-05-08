import glob
import os
import pickle


KIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(KIT_ROOT, "cache")


def find_cache_path(cache_name):
    """Return the path to a cached pickle by basename.

    Public caches are grouped under ``cache/sft`` and
    ``cache/sft_then_rl/<scene>``.  Matching by basename keeps the generated
    shell scripts compact and independent of the nested cache layout.
    """
    direct_path = os.path.join(CACHE_PATH, cache_name)
    if os.path.exists(direct_path):
        return direct_path

    matches = glob.glob(os.path.join(CACHE_PATH, "**", cache_name), recursive=True)
    if not matches:
        raise FileNotFoundError(f"Cache file not found: {cache_name}")
    return sorted(matches)[0]


def load_pickle_by_name(cache_name):
    cache_path = find_cache_path(cache_name)
    print(f"Loading {cache_name} from {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)

