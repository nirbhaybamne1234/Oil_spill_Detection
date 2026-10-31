import numpy as np
from src.metrics import segmentation_metrics

def test_perfect_prediction():
    y=np.array([[0,1],[1,0]])
    m=segmentation_metrics(y,y)
    assert m["dice"] > .999
    assert m["iou"] > .999
    assert m["recall"] > .999
