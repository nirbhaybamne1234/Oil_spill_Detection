import streamlit as st
from PIL import Image
import tempfile, os, io
import numpy as np
from src.infer import predict_image

st.set_page_config(page_title="Oil Spill Detection | SAR",page_icon="🌊",layout="wide")
st.title("🌊 Oil Spill Detection")
st.caption("Pixel-level oil spill segmentation from Sentinel-1 / PALSAR SAR imagery")
st.info("This academic prototype provides decision-support predictions. It is not a substitute for expert review or verified environmental monitoring.")
tab1,tab2=st.tabs(["Detect oil spill","Project information"])
with tab1:
    model=st.selectbox("Select trained model",["UNet","SmallUNet"])
    threshold=st.slider("Segmentation threshold",0.05,0.95,0.50,0.05)
    upload=st.file_uploader("Upload SAR image",type=["png","jpg","jpeg","tif","tiff","bmp"])
    if upload:
        suffix=os.path.splitext(upload.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f:
            f.write(upload.getvalue()); path=f.name
        try:
            prob,mask=predict_image(path,model,threshold)
            col1,col2,col3=st.columns(3)
            original=Image.open(path).convert("L")
            col1.image(original,caption="Input SAR image",use_container_width=True)
            col2.image((mask*255).astype("uint8"),caption="Predicted oil-spill mask",use_container_width=True)
            overlay=np.asarray(original.resize((mask.shape[1],mask.shape[0]))).copy()
            rgb=np.stack([overlay]*3,axis=-1); rgb[mask==1]=[255,40,40]
            col3.image(rgb,caption="Red = predicted oil pixels",use_container_width=True)
            st.metric("Predicted oil pixel proportion",f"{100*mask.mean():.2f}%")
            st.metric("Mean probability",f"{prob.mean():.3f}")
            mask_buffer=io.BytesIO()
            Image.fromarray((mask*255).astype("uint8")).save(mask_buffer,format="PNG")
            st.download_button("Download predicted mask",data=mask_buffer.getvalue(),file_name="predicted_mask.png",mime="image/png")
        except Exception as e: st.error(f"Prediction failed: {e}")
        finally:
            try: os.unlink(path)
            except OSError: pass
with tab2:
    st.subheader("Project overview")
    st.write("The model learns pixel-level patterns in SAR images and predicts a binary mask identifying possible oil-spill regions.")
    st.markdown("""
    **Models compared**
    - Small U-Net: compact encoder-decoder segmentation baseline.
    - U-Net: skip-connected encoder-decoder segmentation network.

    **Evaluation metrics**
    - Dice coefficient
    - Intersection over Union (IoU)
    - Pixel precision and recall
    - Pixel accuracy

    **Dataset:** SOS Deep-SAR Oil Spill Segmentation dataset from Kaggle.
    """)
    st.warning("Predictions are only as reliable as the training data, annotation quality, SAR preprocessing and validation protocol.")
