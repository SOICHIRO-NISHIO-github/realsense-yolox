#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# pythonの実行はanaconda Promptで行う!
# フォルダーはYOLOX-main内で行う! C/YOLOX-main 実行コード↓
# python tools/demo-cor.py image -n yolox-s -c weights/yolox_s.pth --path assets/human.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
# python tools/demo-cor.py video -n yolox-s -c weights/yolox_s.pth --path assets/vtest.avi --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
# python tools/Yolox_motpy2.py webcam -n yolox-s -c weights/yolox_s.pth --path assets/vtest.avi --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
#C:\YOLOX-main\realsense\librealsense-master\wrappers\python

import argparse

import argparse
import os
import sys
sys.path.append(os.getcwd())
import time
from loguru import logger

import cv2
import inspect
import torch
import numpy as np
import numpy as np
import pyrealsense2 as rs
import math

import pandas as pd
import csv

sys.path.append(os.path.abspath(".."))

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from realsense import RealsenseCapture

from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_track

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
__all__ = ["vis"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class MOT:
    def __init__(self):
        self.tracker = MultiObjectTracker(dt=0.1)

    def track(self, outputs, ratio, depth_frame, color_intr):
        if outputs[0] is not None:
        
            outputs = outputs[0].cpu().numpy()
            outputs = [Detection(box=box[:4] / ratio,  score=box[4] * box[5], class_id=box[6]) for box in outputs]
        else:
            outputs = []

        self.tracker.step(detections=outputs)
        tracks = self.tracker.active_tracks()
        return tracks

    #outputsにワールド座標を追加しようとしたが無理だった。
    """"
    def World_Coordinate(self, box, depth_frame, color_intr):
        x = (box[0] + box[2])/2
        y = (box[1] + box[3])/2
        distance = depth_frame.get_distance(int(x),int(y))
        World_cord = rs.rs2_deproject_pixel_to_point(color_intr , [x, y], distance)
        return World_cord
    """



class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, depth_frame, color_intr, cls_conf=0.35):

        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        #print(self)
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        centerlist = []
        
        #推論されたクラス番号を表示
        #print(cls)
        #tensor型をnp.array.flaot型に
        clsarray = np.array(cls)
        #np.array.flaot型をint型に
        clsint = clsarray.astype(int)

        #カウンタ変数
        i = 0

        #print(inspect.getmembers(bboxes))
        for bboxe in bboxes:
                #cnt += 1
            #center = self.centercal(bboxe, center)
            centerlist.append(self.centercal(bboxe, clsint[i]))
            #for cor in bboxe:
            #print("------next------")
            i += 1
            #print(bboxe)
        
        humancenterx = []
        humancentery = []
        bboxe_x1 = []
        bboxe_y1 = []
        bboxe_x2 = []
        bboxe_y2 = []

        humanbboxe = []

    #人と検出された結果のみ、各リストに格納。
        for list in centerlist:
            if list[2] == 0:
                humancenterx.append(list[0])
                humancentery.append(list[1])
                bboxe_x1.append(list[3])
                bboxe_y1.append(list[4])
                bboxe_x2.append(list[5])
                bboxe_y2.append(list[6]) 


        i_bbox = len(bboxe_x1)

    #bboxeの4点の座標を配列に追加
        for i in range(i_bbox):
            humanbboxe.append([bboxe_x1[i], bboxe_y1[i], bboxe_x2[i], bboxe_y1[i], bboxe_x1[i], bboxe_y2[i], bboxe_x2[i], bboxe_y2[i]])


        #print(humancenterx)
        #print(humancentery)

        #クラス番号に該当するクラス名を表示
        #for num in range
        #for cls_nam in clsint:
            #print(cls_nam)
            #print(self.cls_names[cls_nam])
            #if cls_nam == 0:
                #code
        #self.bboxecal(bboxes)

        i = len(humancenterx)
        cnt = i
        k = 0
        f = 0

        #距離関係を視認化
        """
        for a in range(i):
            for j in range(cnt):
                f = j + k
                disx = humancenterx[a] - humancenterx[f]
                disy = humancentery[a] - humancentery[f]
                if disx < 0:
                    disx = disx * -1
                if disy < 0:
                    disy = disy * -1
                dis = (disx + disy) / 2
                dis = int(dis)
                if dis > 255:
                    dis = dis / 10
                cv2.line(img, (int(humancenterx[a]),int(humancentery[a])), (int(humancenterx[f]),int(humancentery[f])), (dis, 0, dis), 1)
            cnt -= 1
            k += 1
        """

        """
        #personのカメラからの距離推定：　推定単位はメートル
        ThreeDimPer = []
        PrintThreeDimPer = 0
        for cor_no in range(i):
            PrintThreeDimPer = depth_frame.get_distance(humancenterx[cor_no], humancentery[cor_no])
            ThreeDimPer.append(depth_frame.get_distance(humancenterx[cor_no], humancentery[cor_no]))
        print(ThreeDimPer)
        """


        i = len(humancenterx)
        cnt = i
        k = 0
        f = 0
        all_est_range = []
        count = 0

        
        #color = (self._COLORS[0] * 255).astype(np.uint8).tolist()

        pair = []

        #人と人との距離を計測。
        for a in range(i):
            for j in range(cnt):
                f = j + k
                ThreeDimPer1 = depth_frame.get_distance(humancenterx[a], humancentery[a])
                point_I = rs.rs2_deproject_pixel_to_point(color_intr , [humancenterx[a], humancentery[a]], ThreeDimPer1)
                ThreeDimPer2 = depth_frame.get_distance(humancenterx[f], humancentery[f])
                point_R = rs.rs2_deproject_pixel_to_point(color_intr , [humancenterx[f], humancentery[f]], ThreeDimPer2)
                est_range = math.sqrt((point_I[0]-point_R[0])*(point_I[0]-point_R[0]) + (point_I[1]-point_R[1])*(point_I[1]-point_R[1]) +(point_I[2]-point_R[2])*(point_I[2]-point_R[2]))
                #print(est_range)
                if est_range < 1 and est_range >0:
                    if len(humanbboxe) >= 2:
                        #print("ok")
                        center_x = (humancenterx[a]+humancenterx[f])/2
                        center_y = (humancentery[a]+humancentery[f])/2
                        radians = self.MaxRadiance(humanbboxe[a], humanbboxe[f], center_x, center_y)
                        #print(radians)
                        pair.append(a)
                        pair.append(f)
                        #cv2.circle(img, (int(center_x), int(center_y)), int(radians), (0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0)
                        #cv2.line(img, (int(humancenterx[a]),int(humancentery[a])), (int(humancenterx[f]),int(humancentery[f])),(0,0,255), 2)
                        #cv2.rectangle(img, (int(humanbboxe[f][0]),int(humanbboxe[f][1])), (int(humanbboxe[f][6]), int(humanbboxe[f][7])), (0,0,255), 2)

                #all_est_range.append(est_range)
                #print(len(humanbboxe))
            cnt -= 1
            k += 1
        #print(all_est_range)

        no_dup_pair = []

        no_dup_pair = set(pair)

        len_human = len(humanbboxe)
        """"
        if len_human >= 2:
            for i in range(len_human):
                if i in no_dup_pair:
                    cv2.rectangle(img, (int(humanbboxe[i][0]),int(humanbboxe[i][1])), (int(humanbboxe[i][6]), int(humanbboxe[i][7])), (0,0,255), 2)
                    cv2.rectangle(img, (humancenterx[i] - 5, humancentery[i] - 5), (humancenterx[i] + 5, humancentery[i] + 5), (0,0,255), -1)
                else:
                    cv2.rectangle(img, (int(humanbboxe[i][0]),int(humanbboxe[i][1])), (int(humanbboxe[i][6]), int(humanbboxe[i][7])), (0,255, 0), 2) 
                    cv2.rectangle(img, (humancenterx[i] - 5, humancentery[i] - 5), (humancenterx[i] + 5, humancentery[i] + 5), (0,255,0), -1)
        """
                

        #cv2.rectangle(img, (int(humancenterx[0]),int(humancentery[0])), (int(humancenterx[1]),int(humancentery[1])), (0, 255, 0), -1)
        vis_res = vis(img, bboxes, scores, humancenterx, humancentery, cls, cls_conf, self.cls_names)
        return vis_res

    #中心点の座標の計算
    def centercal(self, bboxe, clsnum):

        center = []
        cnt = 0
        xcenter = 0
        ycenter = 0

        for cor in bboxe:
            if cnt % 2 == 0:
                xcenter += cor
            else:
                ycenter += cor
            cnt += 1

        xcenter = xcenter / 2
        ycenter = ycenter / 2

        #print(xcenter)
        #print(ycenter)
        x1=np.array(bboxe[0])
        y1=np.array(bboxe[1])
        x2=np.array(bboxe[2])
        y2=np.array(bboxe[3])

        #中心座標とクラス識別結果番号、バウンティングボックスの端っこ座標をlistに格納し、listを返す
        center.append(xcenter) #0
        center.append(ycenter) #1
        center.append(clsnum) #2
        center.append(x1) #3
        center.append(y1) #4
        center.append(x2) #5
        center.append(y2) #6

        return center
        '''
        '''

    
    def MaxRadiance(self, humanbboxe_a, humanbboxe_f, center_x, center_y):

        a = 8
        max = math.sqrt((center_x - humanbboxe_a[0])*(center_x - humanbboxe_a[0]) + (center_y-humanbboxe_a[1])*(center_y-humanbboxe_a[1]))
        for i in range(a):
            if i % 2 == 0:
                tmp1 = math.sqrt((center_x - humanbboxe_a[i])*(center_x - humanbboxe_a[i]) + (center_y-humanbboxe_a[i+1])*(center_y-humanbboxe_a[i+1]))
                if max < tmp1:
                    max = tmp1 
                tmp2 = math.sqrt((center_x - humanbboxe_f[i])*(center_x - humanbboxe_f[i]) + (center_y-humanbboxe_f[i+1])*(center_y-humanbboxe_f[i+1]))
                if max < tmp2:
                    max = tmp2

        return max

    




def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        #print(outputs[0])
        #print()
        #print(predictor.confthre)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        #print(inspect.getmembers(result_image))
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

"""
def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        #cv2.imshow = ("cap", frame)
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            #if args.save_result:
            #vid_writer.write(result_frame)
            #else:
            cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
            cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
"""

def imageflow_demo(predictor, vis_folder, current_time, args):
    mot = MOT()
    #cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    cap = RealsenseCapture() #realsense
    #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    #fps = cap.get(cv2.CAP_PROP_FPS)

    cap.WIDTH = 640
    cap.HEIGHT = 480
    cap.FPS = 30
    
    width = 640
    height = 480
    fps = 30


    #realsense用設定
    #cap2.WIDTH = 640
    #cap2.HEIGHT = 480
    #cap2.FPS = 30
    human_id = []
    #data = [0, 0, 0, 0]
    df = pd.DataFrame(columns=["ID", "X", "Y", "Z"])
    cap.start()
    #df = pd.read_csv("result.csv", encoding = 'shift-jis',header=None)
    #print(df)
    #df.to_csv("result.csv")


    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        RGB_frame = frame[0]
        depth_frame = frame[1]
        color_intr = frame[2] #カメラパラメータ
        #cv2.imshow = ("cap", frame)
        if ret_val:
            outputs, img_info = predictor.inference(RGB_frame)
            result_frame = predictor.visual(outputs[0], img_info, cap.depth_frame, color_intr, predictor.confthre)
            #print(img_info["ratio"])
            #if args.save_result:
            #vid_writer.write(result_frame)
            #else:
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET)
            #cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
            #results_images = np.hstack((result_frame, depth_colormap))
            tracks = mot.track(outputs, img_info["ratio"], cap.depth_frame, color_intr)
            for trc in tracks:
                if trc[3] == 0:
                    id_str = str(trc[0])
                    threecor = Three_Dimensional_Coordinates(trc[1], cap.depth_frame, color_intr)
                    threecor = rotate_x(threecor)
                    #print(threecor)
                    #threecor[0] = x, threecor[1] = y, threecor[2] = z 
                    df = df.append({"ID" : id_str, "X": threecor[0][0], "Y":threecor[0][1],  "Z":threecor[0][2]}, ignore_index = True)
                    #print(id_str, threecor[0][0], threecor[0][1],  threecor[0][2]) デバッグ用
                    df.to_csv("tools/kankeinasi.csv", encoding="utf-8", header=True, index=False)
                    draw_track(result_frame, trc, thickness=1)

            if args.save_result:
                vid_writer.write(result_frame)
                cv2.imshow('frame', result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            #cv2.imshow("yolox", result_frame)
            #cv2.imshow("depth", depth_colormap)
            #cv2.imshow("depth",  results_images)
            #ch = cv2.waitKey(1)
            #if ch == 27 or ch == ord("q") or ch == ord("Q"):
                #break
        else:
            break
    print(df)
    #df.to_csv("tools/test_result2.csv", encoding="utf-8", header=True, index=False)

def Three_Dimensional_Coordinates(box, depth_frame, color_intr):
    x = (box[0] + box[2])/2
    y = (box[1] + box[3])/2
    ThreeDimPer1 = depth_frame.get_distance(int(x), int(y))
    point_I = rs.rs2_deproject_pixel_to_point(color_intr , [x, y], ThreeDimPer1)
    return point_I 

def rotate_x(cor):
    cor = np.array(cor)
    deg = 30
    r = np.radians(deg)
    C = np.cos(r)
    S = np.sin(r)
    R_x = np.matrix((
        (1,0,0),
        (0,C,-S),
        (0,S,C)
    ))
    rotate_cor = np.dot(R_x, cor)
    rotate_cor = rotate_cor.tolist()
    return rotate_cor


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)


