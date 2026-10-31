import torch, numpy as np
from PIL import Image
from src.data import load_gray
from src.models import UNet, SmallUNet
from config import MODEL_DIR, IMAGE_SIZE

def predict_image(path, model_name="UNet", threshold=.5):
    device="cuda" if torch.cuda.is_available() else "cpu"
    cls={"UNet":UNet,"SmallUNet":SmallUNet}[model_name]
    ckpt_path=MODEL_DIR/f"{model_name.lower()}_best.pt"
    if not ckpt_path.exists(): raise FileNotFoundError(f"{ckpt_path} not found. Train the model first.")
    model=cls().to(device); ck=torch.load(ckpt_path,map_location=device); model.load_state_dict(ck["state_dict"]); model.eval()
    arr=load_gray(path,IMAGE_SIZE)
    with torch.no_grad(): prob=torch.sigmoid(model(torch.from_numpy(arr[None,None]).to(device)))[0,0].cpu().numpy()
    return prob, (prob>=threshold).astype(np.uint8)
