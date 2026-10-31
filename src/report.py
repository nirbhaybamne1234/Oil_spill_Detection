"""Generate a basic PDF report using measured evaluation output."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def main():
    p=Path("outputs/val_metrics.json")
    if not p.exists(): raise FileNotFoundError("Run python -m src.evaluate first.")
    metrics=json.loads(p.read_text())
    doc=SimpleDocTemplate("reports/final_project_report.pdf",pagesize=A4)
    Path("reports").mkdir(exist_ok=True); styles=getSampleStyleSheet(); story=[]
    story += [Paragraph("Oil Spill Detection Using Machine Learning",styles["Title"]),
              Paragraph("SAR image semantic segmentation project report",styles["Heading2"]),
              Spacer(1,12),
              Paragraph("Objective: segment possible oil-spill regions in Synthetic Aperture Radar imagery using paired images and binary masks.",styles["BodyText"]),
              Spacer(1,12),Paragraph("Validation results",styles["Heading1"])]
    data=[["Model","Pixel accuracy","Precision","Recall","Dice","IoU"]]
    for name,m in metrics.items():
        data.append([name]+[f"{m[k]:.4f}" for k in ["pixel_accuracy","precision","recall","dice","iou"]])
    table=Table(data,repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324F")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.5,colors.grey),
        ("PADDING",(0,0),(-1,-1),7)]))
    story += [table,Spacer(1,16),Paragraph("Interpretation",styles["Heading1"]),
              Paragraph("Compare models using Dice and IoU as overlap metrics, and recall to assess missed oil pixels. Pixel accuracy can be inflated by the large background area. Replace this concise generated report with a full discussion of dataset provenance, preprocessing, split protocol, limitations and error analysis before academic submission.",styles["BodyText"])]
    doc.build(story); print("Created reports/final_project_report.pdf")
if __name__=="__main__": main()
