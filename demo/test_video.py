#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import sys
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
import cv2
import argparse
import time

def test(stage, profiling):
    print("Start Detecting")
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/pnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = FcnDetector(P_Net,modelPath, profiling) 
    if stage in ['rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/rnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = Detector(R_Net, 24, 1, modelPath, profiling)
    if stage in ['onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/onet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = Detector(O_Net, 48, 1, modelPath, profiling)
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.9, 0.6, 0.7])

    # Now to detect
    camID = 0
    cap = cv2.VideoCapture(camID)
    while True:
        ret, image = cap.read()
        if ret == 0:
            break
        [h, w] = image.shape[:2]
        print (h, w)
        #image_data = cv2.flip(image, 1)
        #image_data = cv2.flip(image, 1)
        image_data = image
        start_time = time.time()
        testImages = []
        testImages.append(image_data)
        allBoxes, allLandmarks = mtcnnDetector.detect_face(testImages)
        inf_time = time.time() - start_time
        print("inference time(s): {}".format(inf_time))
        del testImages[0]
        #print("allBoxes: {}".format(allBoxes))
        #print("allLandmarks: {}".format(allLandmarks))
        #print("\n")
        
        # Save it
        if (len(allBoxes) >=1):
            for idx, bbox in enumerate(allBoxes):
                cv2.putText(image_data,str(np.round(bbox[idx][4],2)),(int(bbox[idx][0]),int(bbox[idx][1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
                cv2.rectangle(image_data, (int(bbox[idx][0]),int(bbox[idx][1])),(int(bbox[idx][2]),int(bbox[idx][3])),(0,0,255))
                allLandmark = allLandmarks[idx][0].tolist()
                total_landmark_pts = len(allLandmark)
                if allLandmark is not None and len(allLandmark) == 10: # pnet and rnet will be ignore landmark
                    for index, landmark in enumerate(allLandmark):
                        for i in range(int(total_landmark_pts/2)):
                            cv2.circle(image_data, (int(allLandmark[2*i]),int(int(allLandmark[2*i+1]))), 3, (255,255,255))
        cv2.imshow('Face/Landmark Detection', image_data)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
    cap.release()
     

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='onet', type=str)
    parser.add_argument('--profile', dest='profile', help='profiling will be enable if true is passed',
                        default='False', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    stage = args.stage
    profiling = False
    if args.profile == 'True':
        profiling = True
    else:
        profiling == False
    print("profiling", profiling)
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # Support stage: pnet, rnet, onet
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # set GPU
    test(stage, profiling)

