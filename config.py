"""
Central configuration for paths and fixed variables.
"""

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = "/data/user_data/yashagar/task_D_D/"
TRAIN_DIR = DATA_ROOT + "training/"
VAL_DIR = DATA_ROOT + "validation/"
# Default data dir used by scripts (training split)
DATA_DIR = TRAIN_DIR

# ─── CALVIN dataset keys ─────────────────────────────────────────────────────
LANG_ANNOTATIONS_SUBDIR = "lang_annotations"
LANG_ANNOTATIONS_FILE = "auto_lang_ann.npy"
IMAGE_KEY = "rgb_static"
ACTION_KEY = "rel_actions"
EPISODE_TEMPLATE = "episode_{:07d}.npz"

# ─── Model defaults ──────────────────────────────────────────────────────────
ACTION_DIM = 7
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 4
MAX_SEQ_LENGTH = 512
RESNET_FEATURE_DIM = 512

# ─── Image preprocessing (CLIP normalization) ────────────────────────────────
IMAGE_SIZE = (224, 224)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ─── Training defaults ───────────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_SEQ_LEN = 128
NUM_WORKERS = 4

# ─── NLP ──────────────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"
