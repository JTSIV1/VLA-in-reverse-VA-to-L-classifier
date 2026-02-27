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
SCENE_OBS_KEY = "scene_obs"
ROBOT_OBS_KEY = "robot_obs"
EPISODE_TEMPLATE = "episode_{:07d}.npz"

# ─── Model defaults ──────────────────────────────────────────────────────────
ACTION_DIM = 7
SCENE_OBS_DIM = 24
ROBOT_OBS_DIM = 15
D_MODEL = 128          # 128 / 8 heads = 16 dims per head (64 was too small at 8 dims/head)
NHEAD = 8
NUM_LAYERS = 4
CROSS_LAYERS = NUM_LAYERS  # how many final layers use cross-modal attention (= NUM_LAYERS means early fusion)
DROPOUT_RATE = 0.1
PATCH_SIZE = 25        # 200/25 = 8x8 = 64 patches per image, balances with ~64-step actions
IMAGE_ENCODER = "scratch"  # "scratch" | "resnet18" | "dinov2" | "r3m"

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

# ─── Pretrained vision encoder ───────────────────────────────────────────
R3M_IMG_SIZE = 224
R3M_VARIANT = "resnet50"  # resnet18 (512-d), resnet34 (512-d), resnet50 (2048-d)

# ─── Action tokenization ───────────────────────────────────────────────────────
BINNING_VOCAB_SIZE = 1024
FAST_TOKENIZER_PATH = "./checkpoints/fast_tokenizer"
VQVAE_TOKENIZER_PATH = "./checkpoints/vqvae_tokenizer"
VQVAE_VOCAB_SIZE = 512

# QueST / OAT tokenizer checkpoints and shared hyperparameters
CHECKPOINT_DIR = "./checkpoints"
QUEST_TOKENIZER_CKPT = "./checkpoints/quest_tokenizer"
OAT_TOKENIZER_CKPT = "./checkpoints/oat_tokenizer"
TOKENIZER_HORIZON = 32          # action chunk length (matches oat/config train_questtok.yaml)
TOKENIZER_DOWNSAMPLE_FACTOR = 4 # QueST temporal downsampling (horizon 32 → 8 latent tokens)
TOKENIZER_FIT_NORM_MAX_TRAJS = 1000  # max trajectories used to fit action normalizer
OAT_NUM_REGISTERS = 8           # OAT register tokens (matches oat/config train_oattok.yaml)

# ─── NLP ──────────────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"
