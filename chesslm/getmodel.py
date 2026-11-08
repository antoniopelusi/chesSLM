import os
import shutil

from huggingface_hub import hf_hub_download

from .config import MODEL_ID, MODEL_PATH, VOCAB_PATH
from .model import read_arch
from .util import log


def run_getmodel():
    for path in (MODEL_PATH, VOCAB_PATH):
        if os.path.exists(path):
            log(f"Already exists, skipping: {path}")
            continue
        log(f"Downloading {path} from {MODEL_ID} ...")
        shutil.copy(hf_hub_download(MODEL_ID, path), path)
        log(f"Saved: {path}")
        if path == MODEL_PATH:
            log(f"Architecture: {read_arch(path)}")
