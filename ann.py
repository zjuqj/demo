import os
import keras
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import frst_alg
import facial_feat_extract
import mlp
from sklearn.utils import shuffle
import pickle
import time

base_path = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "model")
img_data_path = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "detected_images")
label_file_path = os.path.join(os.path.dirname(__file__), "detected_images_labels.txt")
NORM_FAC = 200

def train_model(img_name_idx=False):
    trainX=[]
    trainY=[]
    if os.path.exists(os.path.join(base_path, "train_new_x.npz")):
        trainX = np.load(os.path.join(base_path, "train_new_x.npz"))["arr_0"]
        trainY = np.load(os.path.join(base_path, "train_new_y.npz"))["arr_0"]
    elif img_name_idx:
        with open(label_file_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                img_idx, lb_cls = int(line.split(",")[0]), int(line.split(",")[1])
                feats = facial_feat_extract.extract_facial_features(
                    cascade_classifier,
                    landmark_detector,
                    os.path.join(img_data_path, str(img_idx)+".jpg")
                )
                if not feats or len(feats) < 4:
                    continue
                trainX.append(np.array(feats))
                trainY.append([0,1]) if lb_cls == 1 else trainY.append([1,0])
    else:
        with open(label_file_path, "r") as f:
            for line in f.readlines():
                line=line.strip()
                if len(line) == 0:
                    continue
                img_fname, lb = line.split(",")[0], int(line.split(",")[1])
                feats = facial_feat_extract.extract_facial_features(
                    cascade_classifier,
                    landmark_detector,
                    os.path.join(os.path.dirname(label_file_path), "frames", img_fname)
                )
                if not feats or len(feats) < 4 or not feats[0]:
                    continue
                trainX.append(np.array(feats[:4]))
                trainY.append([0, 1]) if lb==1 else trainY.append([1,0])

        trainX = np.array(trainX, dtype="float32")
        trainY = np.array(trainY, dtype="int32")
        np.savez(os.path.join(base_path, "trainx.npz"), trainX)
        np.savez(os.path.join(base_path, "trainy.npz"), trainY)

    print("trainX shape={}, trainY shape={}".format(trainX.shape, trainY.shape))
    trainX /= trainX.max()
    np.savez("x_train.npz", trainX)
    np.savez("y_train.npz", trainY)
    model = mlp.build_mlp((4,))
    trainX, trainY = shuffle(trainX, trainY)
    model.fit(trainX, trainY, epochs=5, batch_size=1, validation_split=0.2, shuffle=True)
    model.save("fatigue_detect_model.h5")

def util_split_video_frames(video_files, save_dir,step_frames=120):
    frame_count=-1
    idx=0
    for vfile in video_files:
        cap = cv2.VideoCapture(vfile)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count +=1
            if frame_count % step_frames !=0:
                continue
            cv2.imwrite(os.path.join(save_dir,"frame_{:06}.png".format(frame_count)), frame)
            idx +=1

def prepare_yawn_dataset():
    vd_files = os.listdir("../user06/YawDD dataset/Dash/Male")
    vd_files = [os.path.join("../user06/YawDD dataset/Dash/Male", e) for e in vd_files]
    util_split_video_frames(vd_files, "../yawn_dataset/frames")

def prep_fatigue_dataset_train():
    dataset_imgs = []
    trainX = []
    trainY = []
    count = 0
    for img_file in os.listdir(os.path.join(os.path.dirname(__file__), "..", "fatigue_new_dataset","0")):
        dataset_imgs.append(os.path.join(os.path.dirname(__file__), "..", "fatigue_new_dataset","0", img_file))
        ft1,ft2,ft3,ft4,_ = facial_feat_extract.extract_facial_features(cascade_classifier, landmark_detector, dataset_imgs[-1])
        ft1 *= 100
        ft2 *= 100
        ft3 *= 50
        ft4 *= 25
        ft1 = int(ft1) / NORM_FAC
        ft2 = int(ft2) / NORM_FAC
        ft3 = int(ft3) / NORM_FAC
        ft4 = int(ft4) / NORM_FAC
        if ft1 is None:
            continue
        trainX.append([ft1, ft2, ft3, ft4])
        trainY.append([1, 0])
        count +=1
        if count == 320:
            break
    
    count=0
    for img_file in os.listdir(os.path.join(os.path.dirname(__file__), "..", "fatigue_new_dataset","1")):
        dataset_imgs.append(os.path.join(os.path.dirname(__file__), "..", "fatigue_new_dataset","1", img_file))
        ft1,ft2,ft3,ft4,_ = facial_feat_extract.extract_facial_features(cascade_classifier, landmark_detector, dataset_imgs[-1])
        ft1 *= 100
        ft2 *= 100
        ft3 *= 50
        ft4 *= 25
        ft1 = int(ft1) / NORM_FAC
        ft2 = int(ft2) / NORM_FAC
        ft3 = int(ft3) / NORM_FAC
        ft4 = int(ft4) / NORM_FAC
        if ft1 is None:
            continue
        trainX.append([ft1, ft2, ft3, ft4])
        trainY.append([0, 1])
        count +=1
        if count == 320:
            break

    trainX = np.array(trainX, dtype="float32")
    trainY = np.array(trainY, dtype="int32")
    np.savez("train_new_x.npz", trainX)
    np.savez("train_new_y.npz", trainY)
    
    train_model()