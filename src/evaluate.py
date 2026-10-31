import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from src.data import SARSegmentationDataset
from src.models import UNet, SmallUNet
from src.metrics import segmentation_metrics
from config import IMAGE_SIZE, MODEL_DIR, OUTPUT_DIR

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/raw"); p.add_argument("--split",default="val",choices=["val","test"])
    a=p.parse_args(); device="cuda" if torch.cuda.is_available() else "cpu"
    ds=SARSegmentationDataset(a.data,a.split,IMAGE_SIZE)
    dl=DataLoader(ds,batch_size=4,shuffle=False)
    results={}
    for name,cls in [("SmallUNet",SmallUNet),("UNet",UNet)]:
        path=MODEL_DIR/f"{name.lower()}_best.pt"
        if not path.exists(): print("Skipping missing checkpoint:",path); continue
        model=cls().to(device); ck=torch.load(path,map_location=device); model.load_state_dict(ck["state_dict"]); model.eval()
        pp=[]; yy=[]
        with torch.no_grad():
            for x,y in dl: pp.append(torch.sigmoid(model(x.to(device))).cpu().numpy()); yy.append(y.numpy())
        results[name]=segmentation_metrics(np.concatenate(pp),np.concatenate(yy))
    if not results: raise FileNotFoundError("No trained checkpoints found. Run python main.py first.")
    OUTPUT_DIR.mkdir(exist_ok=True); dest=OUTPUT_DIR/f"{a.split}_metrics.json"
    dest.write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=="__main__": main()
