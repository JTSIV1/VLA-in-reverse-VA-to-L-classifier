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
D_MODEL = 128          # 128 / 8 heads = 16 dims per head (64 was too small at 8 dims/head)
NHEAD = 8
NUM_LAYERS = 4
DROPOUT_RATE = 0.1
PATCH_SIZE = 25        # 200/25 = 8x8 = 64 patches per image, balances with ~64-step actions

# ─── Image preprocessing ─────────────────────────────────────────────────────
IMAGE_SIZE = (200, 200) # native CALVIN resolution, no unnecessary upscale
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ─── Training defaults ───────────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_SEQ_LEN = 128
NUM_WORKERS = 4
WARMUP_EPOCHS = 2
GRAD_CLIP_NORM = 1.0

# ─── FAST tokenization ───────────────────────────────────────────────────────
FAST_VOCAB_SIZE = 1024
FAST_TOKENIZER_PATH = "./checkpoints/fast_tokenizer"

# ─── NLP ──────────────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"
