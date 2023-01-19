#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2
import numpy as np
import onnxruntime

class M_LSD:
    def __init__(self, modelpath, conf_thres=0.5, dist_thres=20.0):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath)
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [self.onnx_session.get_outputs()[i].name for i in range(3)]

        self.input_shape = self.onnx_session.get_inputs()[0].shape ### n,h,w,c
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.conf_threshold = conf_thres
        self.dist_threshold = dist_thres

    def prepare_input(self, image):
        resized_image = cv2.resize(image, dsize=(self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        input_image = np.concatenate([resized_image, np.ones([self.input_height, self.input_width, 1])], axis=-1)
        input_image = np.expand_dims(input_image, axis=0).astype('float32')
        return input_image

    def detect(self, image):
        h_ratio, w_ratio = [image.shape[0] / self.input_height, image.shape[1] / self.input_width]
        input_image = self.prepare_input(image)

        # Perform inference on the image
        result = self.onnx_session.run(self.output_names, {self.input_name: input_image})

        pts = result[0][0]
        pts_score = result[1][0]
        vmap = result[2][0]

        start = vmap[:, :, :2]
        end = vmap[:, :, 2:]
        dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

        segments_list = []
        for center, score in zip(pts, pts_score):
            y, x = center
            distance = dist_map[y, x]
            if score > self.conf_threshold and distance > self.dist_threshold:
                disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
                x_start = x + disp_x_start
                y_start = y + disp_y_start
                x_end = x + disp_x_end
                y_end = y + disp_y_end
                segments_list.append([x_start, y_start, x_end, y_end])

        lines = 2 * np.array(segments_list)  # 256 > 512
        lines[:, 0] = lines[:, 0] * w_ratio
        lines[:, 1] = lines[:, 1] * h_ratio
        lines[:, 2] = lines[:, 2] * w_ratio
        lines[:, 3] = lines[:, 3] * h_ratio

        # Draw Line
        dst_image = copy.deepcopy(image)
        for line in lines:
            x_start, y_start, x_end, y_end = [int(val) for val in line]
            cv2.line(dst_image, (x_start, y_start), (x_end, y_end), [0, 0, 255], 3)
        return dst_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/test1.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/model_512x512_large.onnx', help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--distThreshold', default=20.0, type=float, help='dist threshold')
    args = parser.parse_args()

    detector = M_LSD(args.modelpath, conf_thres=args.confThreshold, dist_thres=args.distThreshold)
    srcimg = cv2.imread(args.imgpath)

    dstimg = detector.detect(srcimg)
    cv2.namedWindow('srcimg', 0)
    cv2.imshow('srcimg', srcimg)
    winName = 'Deep learning Line Detect in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()