"""Single source of truth for every experimental setting.

Change values here, never inline in the scripts, so the paper's method
section and the code cannot drift apart.
"""
import os

# Where everything is written. In Colab, point this at Google Drive so a
# disconnect never loses a run:  os.environ["SE_OUT"] = "/content/drive/MyDrive/se_audit_out"
OUT_DIR = os.environ.get("SE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
DATA_DIR = os.path.join(OUT_DIR, "data")

SEED = 0
N_PER_SPLIT = 50

# ---- models -----------------------------------------------------------------
MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi2": "microsoft/phi-2",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # capability-floor case (appendix)
}
MAIN_MODELS = ["qwen1.5b", "qwen3b", "phi2"]
FLOOR_MODELS = ["smollm2"]
MODEL_LABEL = {"qwen1.5b": "Qwen-1.5B", "qwen3b": "Qwen-3B", "phi2": "Phi-2", "smollm2": "SmolLM2-1.7B"}
QWEN_MODELS = ["qwen1.5b", "qwen3b"]

# Sequences per generate() call. Lower it if you hit CUDA OOM on a T4.
GEN_BATCH = {"qwen1.5b": 32, "qwen3b": 24, "phi2": 12, "smollm2": 24}
# Phi-2 has a 2048-token context, so judge prompts truncate each trace harder.
JUDGE_TRACE_TOKENS = {"qwen1.5b": 1024, "qwen3b": 1024, "phi2": 400, "smollm2": 600}

# ---- datasets ---------------------------------------------------------------
SPLITS = ["wol4", "wol5", "gsm3", "gsm5"]
SPLIT_LABEL = {"wol4": "WoL-4", "wol5": "WoL-5", "gsm3": "GSM-3", "gsm5": "GSM-5"}
WOL_SPLITS = ["wol4", "wol5"]

# ---- sampling ---------------------------------------------------------------
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_NEW_TOKENS = 512
K_STYLE = 3   # style-prompted agents
K_SC = 5      # self-consistency samples from one neutral prompt
K_MATCHED = 3  # k-matched self-consistency control (all C(5,3) subsets)

# ---- NLI scorer (ToTh) ------------------------------------------------------
NLI_MODELS = {
    "roberta": "FacebookAI/roberta-large-mnli",
    "bart": "facebook/bart-large-mnli",
}
MAX_STEPS = 40          # cap on reasoning steps per trace
NLI_MAX_LEN = 256
MAIN_MAPPING = "V0"     # trust mapping used for headline ToTh numbers

# ---- trace similarity -------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- analysis ---------------------------------------------------------------
# Configs whose oracle-random gap (in accuracy, 0-1) is below this are flagged
# low-signal with a dagger in the tables. SE is undefined at gap == 0.
LOW_SIGNAL_GAP = 0.15
