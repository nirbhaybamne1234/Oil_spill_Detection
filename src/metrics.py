import numpy as np
def segmentation_metrics(pred, target, threshold=.5, eps=1e-7):
    p = np.asarray(pred) >= threshold
    y = np.asarray(target) >= .5
    tp = np.logical_and(p,y).sum(); fp=np.logical_and(p,~y).sum()
    fn=np.logical_and(~p,y).sum(); tn=np.logical_and(~p,~y).sum()
    precision=tp/(tp+fp+eps); recall=tp/(tp+fn+eps)
    dice=2*tp/(2*tp+fp+fn+eps); iou=tp/(tp+fp+fn+eps)
    accuracy=(tp+tn)/(tp+tn+fp+fn+eps)
    return {"pixel_accuracy":float(accuracy),"precision":float(precision),
            "recall":float(recall),"dice":float(dice),"iou":float(iou)}
