import os
from Evaluator import *
import pdb

def getGTBoxes(cfg, GTFolder):

    files = os.listdir(GTFolder)
    files.sort()

    classes = []
    num_pos = {}
    gt_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(GTFolder, f), "r")
        
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])      
            one_box = [left, top, right, bottom, 0]
              
            if cls not in classes:
                classes.append(cls)
                gt_boxes[cls] = {}
                num_pos[cls] = 0

            num_pos[cls] += 1

            if nameOfImage not in gt_boxes[cls]:
                gt_boxes[cls][nameOfImage] = []
            gt_boxes[cls][nameOfImage].append(one_box)  
            
        fh1.close()
    return gt_boxes, classes, num_pos

def getDetBoxes(cfg, DetFolder):

    files = os.listdir(DetFolder)
    files.sort()

    det_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(DetFolder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            score = float(splitLine[5])
            one_box = [left, top, right, bottom, score, nameOfImage]

            if cls not in det_boxes:
                det_boxes[cls]=[]
            det_boxes[cls].append(one_box)

        fh1.close()
    return det_boxes

def detections(cfg,
               gtFolder,
               detFolder,
               savePath,
               show_process=True):
    

    gt_boxes, classes, num_pos = getGTBoxes(cfg, gtFolder)
    det_boxes = getDetBoxes(cfg, detFolder)
    
    evaluator = Evaluator()

    return evaluator.GetPascalVOCMetrics(cfg, classes, gt_boxes, num_pos, det_boxes)

def plot_save_result(cfg, results, classes, savePath):
    
    
    plt.rcParams['savefig.dpi'] = 80
    plt.rcParams['figure.dpi'] = 130

    acc_AP = 0
    validClasses = 0
    fig_index = 0

    for cls_index, result in enumerate(results):
        if result is None:
            raise IOError('Error: Class %d could not be found.' % classId)

        cls = result['class']
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        acc_AP = acc_AP + average_precision
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        fig_index+=1
        plt.figure(fig_index)
        plt.plot(recall, precision, cfg['colors'][cls_index], label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        ap_str = "{0:.2f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(cls), ap_str))
        plt.legend(shadow=True)
        plt.grid()
        plt.savefig(os.path.join(savePath, cls + '.png'))
        plt.show()
        plt.pause(0.05)


    mAP = acc_AP / fig_index
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    
