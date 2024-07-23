#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image, ImageDraw

from detectron2 import config
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures.masks import BitMasks

import importlib
import maskrcnn_module.custom_roi_heads
importlib.reload(maskrcnn_module
                 .custom_roi_heads)
from maskrcnn_module.custom_roi_heads import ROI_HEADS_REGISTRY, CustomStandardROIHeads
ROI_HEADS_REGISTRY.register(CustomStandardROIHeads)
import maskrcnn_module.custom_dataset_mapper
importlib.reload(maskrcnn_module.custom_dataset_mapper)
from maskrcnn_module.custom_dataset_mapper import CustomDatasetMapper
from maskrcnn_module.utils import *
import maskrcnn_module.custom_predictor
importlib.reload(maskrcnn_module.custom_predictor)
from maskrcnn_module.custom_predictor import CustomPredictor

class Maskrcnn_Module():
    def __init__(self, model_weight_path_1, model_weight_path_2, score_thresh=0.2):
        self.model_weight_path_1 = model_weight_path_1
        self.model_weight_path_2 = model_weight_path_2
        self.path_to_val = 'place_holder.npy'
        self.score_thresh = score_thresh

    ######################################
    # Maskrcnn settings and initialize
    ######################################

    # register dataset
    def get_val_dataset(self):
        return np.load(self.path_to_val, allow_pickle=True)

    '''
    Description: Setup configurations for detectron2 model
    Inputs: first_init is True if init_model is being called for the first time
    Effects: Creates cfg object for model and registers dataset
    Returns: cfg object
    '''
    def init_model(self, first_init=True, model_weights=None):
        # register dataset
        if first_init:
            DatasetCatalog.register("val", self.get_val_dataset)
            # realsense images are flipped left and right from reality so also flip the class names
            MetadataCatalog.get("val").set(thing_classes=["Left-hinged", "Right-hinged", "Bottom-hinged", "Pulls out", "Top-hinged"])
            MetadataCatalog.get("val").set(keypoint_names=["handle_2d"])
            MetadataCatalog.get("val").set(keypoint_flip_map=[("handle_2d", "handle_2d")])

        # setup model
        cfg = config.get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"] 
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.KEYPOINT_ON = True
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 1
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.OUTPUT_DIR = ""
        cfg.DATASETS.TRAIN = ('val',)
        cfg.DATASETS.TEST = ('val',)

        # added cfg options to automate running experiments
        cfg.NUM_FC = 2
        cfg.HANDLE_LOSS = 1.0
        cfg.AXIS_LOSS = 1.0
        cfg.SURFNORM_LOSS = 1.0
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.RANDOM_FLIP = "none"
        cfg.MODEL.ROI_HEADS.NAME = "CustomStandardROIHeads"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.MODEL.AGNOSTIC = False
        cfg.MODEL.CUSTOM_TRAINER = False
        cfg.MODEL.PROJ_HEAD = False
        cfg.COLOR_JITTER_MIN = 0.9
        cfg.COLOR_JITTER_MAX = 1.0
        cfg.COLOR_JITTER_SCALE = 0.1
        cfg.ERROR_HEAD_INPUT_TYPE = 3
        cfg.ERROR_HEAD_OUTPUT_TYPE = 1
        cfg.SURFNORM_OUTPUT_DIM = 1
        cfg.PREDICT_STD = 0
        cfg.QUANTILE_REG = 0

        return cfg

    def create_predictor(self):
        self.cfg = self.init_model()
        self.predictor = CustomPredictor(self.cfg, self.model_weight_path_1, self.model_weight_path_2)

    def run_inference(self, color_frame, depth_frame, camera_intrinsics):
        self.create_predictor()
        
        # Convert images to numpy arrays (if not already)
        if not isinstance(color_frame, np.ndarray):
            color_image = np.asarray(color_frame)
        else:
            color_image = color_frame

        if not isinstance(depth_frame, np.ndarray):
            depth_image = np.asarray(depth_frame)
        else:
            depth_image = depth_frame

        # Apply colormap on depth image
        depth_image_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=-0.04, beta=255.0), cv2.COLORMAP_OCEAN)

        # Adjust color image for visualization
        color_image = np.moveaxis(color_image, 0, 1)
        color_image = np.fliplr(color_image)

        frame = color_image.copy()
        output = self.predictor(frame)

        maskrcnn_predictions = output['instances'].to("cpu").get_fields()
        pred_scores = maskrcnn_predictions['scores']
        pred_classes = maskrcnn_predictions['pred_classes']
        pred_masks = maskrcnn_predictions['pred_masks']
        pred_handles = maskrcnn_predictions['pred_keypoints']
        pred_axis_points = maskrcnn_predictions['pred_axis_points']
        pred_surf_norm = maskrcnn_predictions['pred_surf_norm']
        pred_handle_orientations = maskrcnn_predictions['pred_handle_orientation']

        unique_pred_ind = []
        for i in range(len(pred_scores)):
            pred_handle = pred_handles[i][0].numpy()
            x_2d = pred_handle[0]
            y_2d = pred_handle[1]
            print("i: ", i, " x_2d: ", x_2d, " y_2d: ", y_2d, "class: ", pred_classes[i].item(), "score: ", pred_scores[i])
            row = round(pred_handle[1])
            col = round(pred_handle[0])
            if row < 0 or row >= color_image.shape[1] or col < 0 or col >= color_image.shape[0]:
                continue
            already_used = False
            for j in unique_pred_ind:
                already_used = already_used or pred_masks[j][round(pred_handle[1])][round(pred_handle[0])]
            
            if not already_used and pred_scores[i] >= self.score_thresh:
                unique_pred_ind.append(i)

        print("unique masks: ", unique_pred_ind)

        num_pred = 0
        v = Visualizer(frame[:, :, ::-1], scale=1, instance_mode=ColorMode.IMAGE_BW)
        for i in unique_pred_ind:
            if pred_scores[i] >= self.score_thresh:
                num_pred += 1
                vis = v.overlay_instances(masks=BitMasks(pred_masks[i].unsqueeze(0)), assigned_colors=[(0,0,1)], labels=[MetadataCatalog.get('val').get('thing_classes')[pred_classes[i]]])
        
        polygons = []
        if num_pred > 0:
            pred_img = Image.fromarray(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB))
            draw = ImageDraw.Draw(pred_img)
        for i in unique_pred_ind:
            pred_handle = pred_handles[i][0].numpy()
            x_2d = pred_handle[0]
            y_2d = pred_handle[1]
            if x_2d < 0 or y_2d < 0 or round(x_2d) >= color_image.shape[1] or round(y_2d) >= color_image.shape[0]:
                continue
            depth = depth_image[round(y_2d), round(x_2d)]
            if depth <= 0.0:
                continue

            if pred_scores[i] >= self.score_thresh:
                width, height = pred_img.size
                scale_factor = 1 / 200
                circle_width = width * scale_factor
                pred_handle = pred_handles[i][0]
                draw.ellipse([pred_handle[0] - circle_width, pred_handle[1] - circle_width, pred_handle[0] + circle_width, pred_handle[1] + circle_width], fill=(0, 255, 0))
                draw.text((pred_handle[0], pred_handle[1] - 10 * circle_width), f"{i}", (0, 0, 255))

                contours, heiarchy = cv2.findContours(pred_masks[i].numpy().astype(np.ubyte), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hull = cv2.convexHull(contours[0])
                perim = cv2.arcLength(hull, True)
                poly = cv2.approxPolyDP(hull, 0.02 * perim, True)
                polygons.append(poly)
                print("i: ", i, "polygon: ", poly.shape)

                pred_img_arr = np.asarray(pred_img).copy()
                cv2.drawContours(pred_img_arr, [poly], 0, (0, 0, 255), 1)

                pred_img = Image.fromarray(pred_img_arr)
                draw = ImageDraw.Draw(pred_img)

                print(f"Handle orientation of {i}'th object:", pred_handle_orientations[i])

        output_dict = []
        for i in unique_pred_ind:
            pred_handle = pred_handles[i][0].numpy()
            x_2d = round(color_image.shape[1] - pred_handle[0] - 1)
            y_2d = round(pred_handle[1])
            depth = depth_image[y_2d, x_2d]
            result = self.deproject_pixel_to_point(camera_intrinsics, [y_2d, x_2d], depth)
            y_3d = result[0]
            x_3d = result[1]

            poly = polygons[unique_pred_ind.index(i)]

            lateral_diff = abs(poly[0][0][0] - poly[1][0][0])
            height_diff = abs(poly[0][0][1] - poly[1][0][1])
            if lateral_diff > height_diff:
                poly = np.append(poly, [poly[0]], axis=0)[1:]

            for j in range(len(poly)):
                poly[j][0][0] = color_image.shape[1] - poly[j][0][0] - 1

            poly_3d = []
            for point in poly:
                point_y = point[0][1]
                point_x = point[0][0]
                point_depth = depth_image[point_y, point_x]
                point_3d = self.deproject_pixel_to_point(camera_intrinsics, [point_y, point_x], point_depth)
                poly_3d.append(np.array([point_3d[1], point_3d[0], point_depth]))

            pred_class = pred_classes[i].numpy()
            class_names = ["Left-hinged", "Right-hinged", "Bottom-hinged", "Pulls out", "Top-hinged"]

            output_dict.append({
                'handle_3d': np.array([x_3d, y_3d, depth]),
                'classification': class_names[pred_class],
                'polygon': np.array(poly_3d),
                'handle_orientation': pred_handle_orientations[i].numpy()
            })

        return output_dict

    def deproject_pixel_to_point(self, intrinsics, pixel, depth):
        x = (pixel[0] - intrinsics[0]) * depth / intrinsics[2]
        y = (pixel[1] - intrinsics[1]) * depth / intrinsics[3]
        return np.array([x, y, depth])

    def deproject_pixel_to_point_plane(self, x_2d, y_2d, depth_image_averaged, pred_mask):
        points = []
        for y in range(pred_mask.shape[0]):
            for x in range(pred_mask.shape[1]):
                if pred_mask[y, x]:
                    z = depth_image_averaged[y, x]
                    if z > 0:
                        points.append([x, y, z])
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        cov_matrix = np.dot(points.T, points)
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
        normal = eig_vecs[:, np.argmin(eig_vals)]
        return np.dot(points, normal) + centroid, centroid, normal

    def run_inference_preview(self, color_frame, depth_frame, camera_intrinsics):
        self.create_predictor()
        
        frame_count_color = 0
        print("Press Ctrl-C to start motion planning")
        maskrcnn_predictions = None
        try:
            # Convert images to numpy arrays (if not already)
            if not isinstance(color_frame, np.ndarray):
                color_image = np.asarray(color_frame)
            else:
                color_image = color_frame

            if not isinstance(depth_frame, np.ndarray):
                depth_image = np.asarray(depth_frame)
            else:
                depth_image = depth_frame

            # Apply colormap on depth image
            depth_image_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=-0.04, beta=255.0), cv2.COLORMAP_OCEAN)

            # Adjust color image for visualization
            color_image = np.moveaxis(color_image, 0, 1)
            color_image = np.fliplr(color_image)

            frame = color_image.copy()
            output = self.predictor(frame)

            maskrcnn_predictions = output['instances'].to("cpu").get_fields()
            pred_scores = maskrcnn_predictions['scores']
            pred_classes = maskrcnn_predictions['pred_classes']
            pred_masks = maskrcnn_predictions['pred_masks']
            pred_handles = maskrcnn_predictions['pred_keypoints']
            pred_axis_points = maskrcnn_predictions['pred_axis_points']
            pred_surf_norm = maskrcnn_predictions['pred_surf_norm']
            pred_handle_orientations = maskrcnn_predictions['pred_handle_orientation']

            unique_pred_ind = []
            for i in range(len(pred_scores)):
                pred_handle = pred_handles[i][0].numpy()
                x_2d = pred_handle[0]
                y_2d = pred_handle[1]
                print("i: ", i, " x_2d: ", x_2d, " y_2d: ", y_2d, "class: ", pred_classes[i].item(), "score: ", pred_scores[i])
                row = round(pred_handle[1])
                col = round(pred_handle[0])
                if row < 0 or row >= color_image.shape[1] or col < 0 or col >= color_image.shape[0]:
                    continue
                already_used = False
                for j in unique_pred_ind:
                    already_used = already_used or pred_masks[j][round(pred_handle[1])][round(pred_handle[0])]
                
                if not already_used and pred_scores[i] >= self.score_thresh:
                    unique_pred_ind.append(i)

            print("unique masks: ", unique_pred_ind)

            num_pred = 0
            v = Visualizer(frame[:, :, ::-1], scale=1, instance_mode=ColorMode.IMAGE_BW)
            for i in unique_pred_ind:
                if pred_scores[i] >= self.score_thresh:
                    num_pred += 1
                    vis = v.overlay_instances(masks=BitMasks(pred_masks[i].unsqueeze(0)), assigned_colors=[(0,0,1)], labels=[MetadataCatalog.get('val').get('thing_classes')[pred_classes[i]]])
            
            polygons = []
            if num_pred > 0:
                pred_img = Image.fromarray(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB))
                draw = ImageDraw.Draw(pred_img)
            for i in unique_pred_ind:
                pred_handle = pred_handles[i][0].numpy()
                x_2d = pred_handle[0]
                y_2d = pred_handle[1]
                if x_2d < 0 or y_2d < 0 or round(x_2d) >= color_image.shape[1] or round(y_2d) >= color_image.shape[0]:
                    continue
                depth = depth_image[round(y_2d), round(x_2d)]
                if depth <= 0.0:
                    continue

                if pred_scores[i] >= self.score_thresh:
                    width, height = pred_img.size
                    scale_factor = 1/200
                    circle_width = width * scale_factor
                    pred_handle = pred_handles[i][0]
                    draw.ellipse([pred_handle[0] - circle_width, pred_handle[1] - circle_width, pred_handle[0] + circle_width, pred_handle[1] + circle_width], fill=(0,255,0))
                    draw.text((pred_handle[0], pred_handle[1] - 10 * circle_width), f"{i}", (0, 0, 255))

                    contours, heiarchy = cv2.findContours(pred_masks[i].numpy().astype(np.ubyte), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    hull = cv2.convexHull(contours[0])
                    perim = cv2.arcLength(hull, True)
                    poly = cv2.approxPolyDP(hull, 0.02 * perim, True)
                    polygons.append(poly)
                    print("i: ", i, "polygon: ", poly.shape)

                    pred_img_arr = np.asarray(pred_img).copy()
                    cv2.drawContours(pred_img_arr, [poly], 0, (0, 0, 255), 1)

                    pred_img = Image.fromarray(pred_img_arr)
                    draw = ImageDraw.Draw(pred_img)

                    print(f"Handle orientation of {i}'th object:", pred_handle_orientations[i])

            cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
            if num_pred > 0:
                cv2.imshow('Realsense', np.array(pred_img))
            else:
                cv2.imshow('Realsense', frame)
            if cv2.waitKey(1) & 0xFF != 255:
                raise KeyboardInterrupt()
                    
            print(f"Number of Predictions for Frame {frame_count_color}: {num_pred}")
            frame_count_color += 1

        except KeyboardInterrupt:
            cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
            if num_pred > 0:
                cv2.imshow('Realsense', np.array(pred_img))
            else:
                cv2.imshow('Realsense', frame)
            user_input = int(input("Select prediction: "))
            pred_handles = maskrcnn_predictions['pred_keypoints']
            if len(pred_handles) > 0 and user_input < len(pred_handles):
                pred_class = maskrcnn_predictions['pred_classes'][user_input].numpy()
                pred_handle_orientation = maskrcnn_predictions['pred_handle_orientation'][user_input].numpy()
                pred_handle_orientation = 'vertical' if pred_handle_orientation[2] > pred_handle_orientation[3] else 'horizontal'
                
                class_names = ["Left-hinged", "Right-hinged", "Bottom-hinged", "Pulls out", "Top-hinged"]
                
                pred_handle = pred_handles[user_input][0].numpy()
                x_2d = round(color_image.shape[1] - pred_handle[0] - 1)
                y_2d = round(pred_handle[1])
                depth = depth_image[y_2d, x_2d]
                result = self.deproject_pixel_to_point(camera_intrinsics, [y_2d, x_2d], depth)
                y_3d = result[0]
                x_3d = result[1]

                poly = polygons[unique_pred_ind.index(user_input)]

                lateral_diff = abs(poly[0][0][0] - poly[1][0][0])
                height_diff = abs(poly[0][0][1] - poly[1][0][1])
                if lateral_diff > height_diff:
                    poly =  np.append(poly, [poly[0]], axis=0)[1:] 
                
                for i in range(len(poly)):
                    poly[i][0][0] = color_image.shape[1] - poly[i][0][0] - 1

                poly_3d = []
                for point in poly:
                    point_y = point[0][1]
                    point_x = point[0][0]
                    point_depth = depth_image[point_y, point_x]
                    point_3d = self.deproject_pixel_to_point(camera_intrinsics, [point_y, point_x], point_depth)
                    poly_3d.append(np.array([point_3d[1], point_3d[0], point_depth]))

                depth_image_averaged = np.zeros((1, color_image.shape[1], color_image.shape[0]))
                depth_image = np.moveaxis(depth_image, 0, 1)
                depth_image_averaged[0] = depth_image
                depth_image_averaged = np.average(depth_image_averaged, axis=0)
                pred_mask = maskrcnn_predictions['pred_masks'][user_input].numpy().astype(np.uint8)
                pred_mask = np.fliplr(pred_mask)
                pred_handle_3d_lookup, pred_handle_3d_plane, normal = self.deproject_pixel_to_point_plane(x_2d, y_2d, depth_image_averaged, pred_mask)

                normal_corrected = np.array([normal[0], -normal[1], -normal[2]])

                handle_3d = np.array([x_3d, y_3d, depth])
                handle_3d_plus_surfnorm = handle_3d + normal_corrected 
                output_dict = {'handle_3d': np.array([x_3d, y_3d, depth]),
                                'handle_3d_plus_surfnorm': handle_3d_plus_surfnorm,
                                'classification': class_names[pred_class],
                                'polygon': np.array(poly_3d),
                                'handle_orientation': pred_handle_orientation}
                print(1)
            else:
                print("No prediction selected. Exiting")
                output_dict = None            
                print(2)
            cv2.destroyAllWindows()

            
            return output_dict