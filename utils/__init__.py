def accuracy_gpu(pred, truth):
    agreeing = pred.eq(truth)
    acc = agreeing.sum().float()/agreeing.numel()
    return float(acc)