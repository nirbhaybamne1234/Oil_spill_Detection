"""Project entry point: dataset audit and model training."""
import argparse
from pathlib import Path
from src.data import find_pairs
from src.train import main as train_main

def audit(data):
    print("Dataset root:",Path(data).resolve())
    for split in ("train","val","test"):
        try:
            pairs=find_pairs(data,split)
            print(f"{split}: {len(pairs)} image-mask pairs")
        except FileNotFoundError:
            if split in ("train","val"): raise
            print(f"{split}: not found (optional)")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/raw")
    p.add_argument("--epochs",type=int,default=20)
    args=p.parse_args()
    audit(args.data)
    import sys
    sys.argv=["train.py","--data",args.data,"--epochs",str(args.epochs)]
    train_main()
