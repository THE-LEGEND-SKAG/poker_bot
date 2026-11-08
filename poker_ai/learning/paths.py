"""File locations shared by training (needs PyTorch) and play (NumPy only)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[1] / "data"
POLICY_PATH = DATA / "neural_policy.npz"
LOG_PATH = DATA / "neural_training_log.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
