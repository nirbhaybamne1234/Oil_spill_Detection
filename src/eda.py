"""Create a lightweight inventory and sample visualizations for paired SAR data."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.data import find_pairs, load_gray
from config import OUTPUT_DIR, IMAGE_SIZE

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data/raw"); a=p.parse_args()
    out=OUTPUT_DIR/"eda"; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ["train","val","test"]:
        try: pairs=find_pairs(a.data,split)
        except FileNotFoundError: continue
        for i,(ip,mp) in enumerate(pairs):
            x=load_gray(ip,IMAGE_SIZE); y=load_gray(mp,IMAGE_SIZE,True)
            rows.append({"split":split,"image":str(ip),"mask":str(mp),"width":Image.open(ip).width,
                         "height":Image.open(ip).height,"oil_pixel_fraction":float(y.mean()),
                         "image_mean":float(x.mean()),"image_std":float(x.std())})
        # Sample grid, up to 8 pairs
        sample=pairs[:8]
        fig,axes=plt.subplots(len(sample),2,figsize=(7,max(2,2*len(sample))))
        if len(sample)==1: axes=np.array([axes])
        for ax,(ip,mp) in zip(axes,sample):
            ax[0].imshow(load_gray(ip),cmap="gray"); ax[0].set_title(ip.name); ax[0].axis("off")
            ax[1].imshow(load_gray(mp,mask=True),cmap="gray"); ax[1].set_title("Ground-truth mask"); ax[1].axis("off")
        fig.tight_layout(); fig.savefig(out/f"{split}_sample_grid.png",dpi=140); plt.close(fig)
    if not rows: raise RuntimeError("No image/mask pairs discovered.")
    with (out/"image_inventory.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} image records to {out}")
if __name__=="__main__": main()
