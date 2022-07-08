from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import confusion_matrix as tm_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

eval_metrics = ['prec','recall','f1','specificity']
eval_metrics.extend(['tn','fp','fn','tp'])
eval_metrics.extend(['auroc','auprc'])
eval_metrics.extend(['auprcf1'])

def calculate_any_metrics(target, metrics, preds=None, probs=None, logits=None, 
    num_classes=None, n_round=2, threshold=0.5):
    assert all(m in ['prec','recall','f1','f2','specificity',
                     'acc','correct','n_total',
                     'tn','fp','fn','tp',
                     'auroc','auprc',
                     'probs','preds',
                    'auprcf1'] for m in metrics), 'undefined metric name'
    if logits is not None:
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=float)
        logits = logits.detach()
        logits = logits.view(-1)
    if isinstance(target, np.ndarray):
        target = torch.tensor(target.copy(), dtype=float)
    target = target.view(-1, 1)
    target = (target.detach()).int()
    results = {}
    # logits to probs and preds
    if logits is not None:
        probs, preds = transform_logit(logits=logits, target=target)
        p_metrics = [m for m in metrics if m in ['probs','preds']]
        for m in p_metrics:
            results[m] = eval(m)
            
    if probs is not None and preds is None:
        if isinstance(probs, np.ndarray):
            probs = torch.tensor(probs, dtype=float)
            probs = probs.detach()
        preds = (probs>threshold).long()
        if target is not None:
            preds = preds.reshape(target.shape) #torch.Size([64, 1])

    # confusion matrix
    tn, fp, fn, tp = calculate_confmat(target=target, preds=preds)
    
    # aucs
    auc_metrics = [m for m in metrics if m in ['auroc','auprc']]
    if auc_metrics:
        results.update(calculate_aucs(target, probs, auc_metrics))
    
    # the others
    other_metrics = [m for m in metrics if m not in list(results.keys()) and m!='auprcf1']
    if other_metrics:
        results.update(calculate_metrics(tn, fp, fn, tp, other_metrics))
    
    # auprc_f1
    results['auprcf1'] = np.nansum([results[m] for m in results if m in ['auprc', 'f1']])
    
    if n_round is not None:
        results = {k: round(results[k], 2) if isinstance(results[k], float) else results[k] for k in results}
    return results
    
def transform_logit(logits, target=None, prt_shape=False):
    if prt_shape: print('logits', logits.shape)
        
    # logits2probs
    if len(logits.shape)>1 and logits.shape[1]>1: # torch.Size([64, 2])
        probs = F.softmax(logits, dim=1)[:,1]
    if len(logits.shape)>1 and logits.shape[1]==1: # torch.Size([64, 1])
        probs = torch.sigmoid(logits)[:,0]        
    elif len(logits.shape)==1: # torch.Size([64])
        probs = torch.sigmoid(logits)

    if prt_shape: print('probs', probs.shape) # torch.Size([64])
    
    #probs2preds
    preds = (probs>0.5).long() # torch.Size([64])
    
    if target is not None:
        preds = preds.reshape(target.shape) #torch.Size([64, 1])
    return probs, preds

def calculate_confmat(target, preds=None, probs=None, num_classes=2):
    if probs is not None:
        preds = (probs>0.5).long()

    if isinstance(target, np.ndarray) or isinstance(target, list):
        tn, fp, fn, tp = confusion_matrix(target, preds).ravel()
    
    if torch.is_tensor(target):
        if num_classes is None:
            num_classes = max(target.shape[1] if len(target.shape)==2 else 0, len(target.unique()))
    
        tn, fp, fn, tp = torch.flatten(tm_confusion_matrix(preds, target, num_classes=num_classes)).cpu().numpy()
    return tn, fp, fn, tp

def calculate_metrics(tn, fp, fn, tp, metrics):
    assert all(m in ['prec','recall','f1','f2', 'specificity',
                     'acc','correct','n_total',
                     'tn','fp','fn','tp'] for m in metrics), 'undefined metric name'
    tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
    prec = (tp/(tp+fp)) if (tp+fp)>0 else 1e-5
    recall = tp/(tp+fn) if (tp+fn)>0 else 1e-5
    specificity = tn/(tn+fp) if (tn+fp)>0 else 1e-5  
    f1 = 2*prec*recall/(prec+recall+1e-5)
    f2 = 5*prec*recall/(4*prec+recall+1e-5)
    n_total = tn+fp+fn+tp
    correct = tp+tn
    acc = correct/n_total
    
    results = {}
    for m in metrics:
        results[m] = eval(m)  
    return results

def calculate_aucs(target, probs, metrics):
    assert all(m in ['auroc','auprc'] for m in metrics), 'undefined metric name'
    target = target.cpu()
    probs = probs.cpu()
    if 'auroc' in metrics:
        try:
            auroc = roc_auc_score(target, probs)
        except Exception as e:
            print(e)
            auroc = np.nan
    if 'auprc' in metrics:
        try:
            auprc = average_precision_score(target, probs)
        except Exception as e:
            print(e)
            auprc = np.nan
    results = {}
    for m in metrics:
        results[m] = eval(m)     
    return results


def plot_roc_curve(test_y, prob, path):
    print(test_y,prob)
    fper, tper, _ = roc_curve(test_y, prob)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(path, dpi=300)

