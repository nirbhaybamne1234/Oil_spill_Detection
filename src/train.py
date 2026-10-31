import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from src.data import SARSegmentationDataset
from src.models import UNet, SmallUNet
from src.metrics import segmentation_metrics

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device))
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.numpy())
    return segmentation_metrics(np.concatenate(ps),np.concatenate(ys))

def train_one(name, model, train_loader, val_loader, epochs, device, outdir):
    model.to(device); opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    criterion=nn.BCEWithLogitsLoss()
    best=-1; history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for x,y in tqdm(train_loader,desc=f"{name} {epoch+1}/{epochs}",leave=False):
            x,y=x.to(device),y.to(device); opt.zero_grad()
            loss=criterion(model(x),y); loss.backward(); opt.step(); losses.append(loss.item())
        metrics=evaluate(model,val_loader,device)
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),**metrics})
        if metrics["dice"]>best:
            best=metrics["dice"]; torch.save({"state_dict":model.state_dict(),"model":name,"image_size":IMAGE_SIZE},outdir/f"{name.lower()}_best.pt")
        print(f"{name} epoch {epoch+1}: loss={np.mean(losses):.4f}, val Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
    (outdir/f"{name.lower()}_history.json").write_text(json.dumps(history,indent=2))
    return history[-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default=str(DATA_DIR/"raw"))
    ap.add_argument("--epochs",type=int,default=EPOCHS)
    ap.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    args=ap.parse_args(); seed_everything()
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:",device)
    train_ds=SARSegmentationDataset(args.data,"train",IMAGE_SIZE,True)
    val_ds=SARSegmentationDataset(args.data,"val",IMAGE_SIZE,False)
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    MODEL_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)
    results={}
    for name,model in [("SmallUNet",SmallUNet()),("UNet",UNet())]:
        results[name]=train_one(name,model,train_dl,val_dl,args.epochs,device,MODEL_DIR)
    (OUTPUT_DIR/"model_comparison.json").write_text(json.dumps(results,indent=2))
    print("Validation comparison:",json.dumps(results,indent=2))
    print("Best model by validation Dice:",max(results,key=lambda k:results[k]["dice"]))
if __name__=="__main__": main()
