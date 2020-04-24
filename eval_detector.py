import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    box_1_width = box_1[2] - box_1[0]
    box_2_width = box_2[2] - box_2[0]
    box_1_height = box_1[3] - box_1[1]
    box_2_height = box_2[3] - box_2[1]

    x = sorted([box_1[1], box_1[3], box_2[1], box_2[3]])
    y = sorted([box_1[0], box_1[2], box_2[0], box_2[2]])

    if (((x[3]-x[0]) > (box_1_width + box_2_width)) or ((y[3]-y[0]) > (box_1_height + box_2_height))):
        iou = 0
    else: 
        i = (x[2]-x[1])*(y[2]-y[1])
        box_1_area = box_1_width * box_1_height
        box_2_area = box_2_width * box_2_height
        u = box_1_area + box_2_area - i 
        iou = i / u 

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0  
    '''
    BEGIN YOUR CODE
    '''

    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            correct_box = 0 
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                conf = pred[j][4]
                if iou >= iou_thr and conf > conf_thr:
                    TP = TP + 1
                    correct_box = correct_box + 1  
                elif iou < iou_thr and conf > conf_thr:
                    FP = FP + 1
            FN = len(gt) - correct_box 


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../../data/hw02_preds'
gts_path = '../../data/hw02_annotations'

# load splits:
split_path = '../../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 
confi_list = np.array([], dtype=float)
for fname in preds_train:
    confi_list = np.concatenate((confi_list, (np.array((preds_train[fname]), dtype=float)[:, 4])))
confidence_thrs = np.sort(confi_list) # using (ascending) list of confidence scores as thresholds
# confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds


tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
precision_set0 = np.zeros(len(confidence_thrs))
recall_set0 = np.zeros(len(confidence_thrs))
precision_set1 = np.zeros(len(confidence_thrs))
recall_set1 = np.zeros(len(confidence_thrs))
precision_set2 = np.zeros(len(confidence_thrs))
recall_set2 = np.zeros(len(confidence_thrs))

for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.25, conf_thr=conf_thr)
    precision_set0[i]  = tp_train[i] / (tp_train[i] + fp_train[i])
    recall_set0[i]  = tp_train[i] / (tp_train[i] + fn_train[i])

for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)
    precision_set1[i]  = tp_train[i] / (tp_train[i] + fp_train[i])
    recall_set1[i]  = tp_train[i] / (tp_train[i] + fn_train[i])

for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.75, conf_thr=conf_thr)
    precision_set2[i]  = tp_train[i] / (tp_train[i] + fp_train[i])
    recall_set2[i]  = tp_train[i] / (tp_train[i] + fn_train[i])

# print(precision)
# print(recall)
# Plot training set PR curves
plt.plot(recall_set0, precision_set0, label='iou=0.25')
plt.plot(recall_set1, precision_set1, label='iou=0.5')
plt.plot(recall_set2, precision_set2, label='iou=0.75')
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('training set')
plt.legend()
plt.show()

if done_tweaking:
    print('Code for plotting test set PR curves.')

    # For a fixed IoU threshold, vary the confidence thresholds.
    # The code below gives an example on the training set for one IoU threshold. 
    confi_list = np.array([], dtype=float)
    for fname in preds_test:
        confi_list = np.concatenate((confi_list, (np.array((preds_test[fname]), dtype=float)[:, 4])))
    confidence_thrs = np.sort(confi_list) # using (ascending) list of confidence scores as thresholds
    # confidence_thrs = np.sort(np.array([preds_test[fname][4] for fname in preds_test],dtype=float)) # using (ascending) list of confidence scores as thresholds

    # '''

    tp_test = np.zeros(len(confidence_thrs))
    fp_test = np.zeros(len(confidence_thrs))
    fn_test = np.zeros(len(confidence_thrs))
    precision_set0 = np.zeros(len(confidence_thrs))
    recall_set0 = np.zeros(len(confidence_thrs))
    precision_set1 = np.zeros(len(confidence_thrs))
    recall_set1 = np.zeros(len(confidence_thrs))
    precision_set2 = np.zeros(len(confidence_thrs))
    recall_set2 = np.zeros(len(confidence_thrs))

    for i, conf_thr in enumerate(confidence_thrs):
        tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=0.25, conf_thr=conf_thr)
        precision_set0[i]  = tp_test[i] / (tp_test[i] + fp_test[i])
        recall_set0[i]  = tp_test[i] / (tp_test[i] + fn_test[i])

    for i, conf_thr in enumerate(confidence_thrs):
        tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=0.5, conf_thr=conf_thr)
        precision_set1[i]  = tp_test[i] / (tp_test[i] + fp_test[i])
        recall_set1[i]  = tp_test[i] / (tp_test[i] + fn_test[i])

    for i, conf_thr in enumerate(confidence_thrs):
        tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=0.75, conf_thr=conf_thr)
        precision_set2[i]  = tp_test[i] / (tp_test[i] + fp_test[i])
        recall_set2[i]  = tp_test[i] / (tp_test[i] + fn_test[i])

    # print(precision)
    # print(recall)
    # Plot training set PR curves
    plt.plot(recall_set0, precision_set0, label='iou=0.25')
    plt.plot(recall_set1, precision_set1, label='iou=0.5')
    plt.plot(recall_set2, precision_set2, label='iou=0.75')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('testing set')
    plt.legend()
    plt.show()
