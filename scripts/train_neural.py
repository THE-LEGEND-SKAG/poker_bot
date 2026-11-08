"""Train the Level 5 neural agent with PPO against a league.

usage: python scripts/train_neural.py [minutes] [workers]
Writes poker_ai/data/neural_policy.npz (NumPy weights used at play time),
poker_ai/data/neural_training_log.json (curves + evaluations) and
checkpoints/ppo_*.pt (PyTorch state dicts, not committed).
"""

import json
import sys

from poker_ai.learning.ppo import train

if __name__ == "__main__":
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    meta = train(minutes=minutes, workers=workers, log=lambda m: print(m, flush=True))
    print(json.dumps(meta, indent=1))
