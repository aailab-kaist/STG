import os
import _io
import math
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops
import onnxruntime

__labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


def _read_image(image_path, target_size=320):
    if isinstance(image_path, str):
        mat = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        mat = image_path
    elif isinstance(image_path, bytes):
        mat = cv2.imdecode(np.frombuffer(image_path, np.uint8), -1)
    elif isinstance(image_path, _io.BufferedReader):
        mat = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), -1)
    else:
        raise ValueError(
            "please make sure the image_path is str or np.ndarray or bytes"
        )

    image_original_width, image_original_height = mat.shape[1], mat.shape[0]

    mat_c3 = cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)

    max_size = max(mat_c3.shape[:2])
    x_pad = max_size - mat_c3.shape[1]
    x_ratio = max_size / mat_c3.shape[1]
    y_pad = max_size - mat_c3.shape[0]
    y_ratio = max_size / mat_c3.shape[0]

    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    input_blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,
        (target_size, target_size),
        (0, 0, 0),
        swapRB=True,
        crop=False,
    )

    return (
        input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


def _postprocess(
    output,
    x_pad,
    y_pad,
    x_ratio,
    y_ratio,
    image_original_width,
    image_original_height,
    model_width,
    model_height,
):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0:4]

            x = x - w / 2
            y = y - h / 2

            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height

            x = x
            y = y

            x = max(0, min(x, image_original_width))
            y = max(0, min(y, image_original_height))
            w = min(w, image_original_width - x)
            h = min(h, image_original_height - y)

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        x, y, w, h = box
        detections.append(
            {
                "class": __labels[class_id],
                "score": float(score),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )

    return detections


class NudeDetector:
    def __init__(self, model_path=None, providers=None, inference_resolution=320):
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "320n.onnx")
            if not model_path
            else model_path,
        )
        model_inputs = self.onnx_session.get_inputs()

        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.input_name = model_inputs[0].name

    def detect(self, image_path):
        (
            preprocessed_image,
            x_ratio,
            y_ratio,
            x_pad,
            y_pad,
            image_original_width,
            image_original_height,
        ) = _read_image(image_path, self.input_width)
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(
            outputs,
            x_pad,
            y_pad,
            x_ratio,
            y_ratio,
            image_original_width,
            image_original_height,
            self.input_width,
            self.input_height,
        )

        return detections

    def detect_batch(self, image_paths, batch_size=4):
        all_detections = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            batch_inputs = []
            batch_metadata = []

            for image_path in batch:
                (
                    preprocessed_image,
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = _read_image(image_path, self.input_width)
                batch_inputs.append(preprocessed_image)
                batch_metadata.append(
                    (
                        x_ratio,
                        y_ratio,
                        x_pad,
                        y_pad,
                        image_original_width,
                        image_original_height,
                    )
                )

            batch_input = np.vstack(batch_inputs)

            outputs = self.onnx_session.run(None, {self.input_name: batch_input})

            for j, metadata in enumerate(batch_metadata):
                (
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = metadata
                detections = _postprocess(
                    [outputs[0][j : j + 1]],
                    x_pad,
                    y_pad,
                    x_ratio,
                    y_ratio,
                    image_original_width,
                    image_original_height,
                    self.input_width,
                    self.input_height,
                )
                all_detections.append(detections)

        return all_detections

    def censor(self, image_path, classes=[], output_path=None):
        detections = self.detect(image_path)
        if classes:
            detections = [
                detection for detection in detections if detection["class"] in classes
            ]

        img = cv2.imread(image_path)

        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            img[y : y + h, x : x + w] = (0, 0, 0)

        if not output_path:
            image_path, ext = os.path.splitext(image_path)
            output_path = f"{image_path}_censored{ext}"

        cv2.imwrite(output_path, img)

        return output_path

class NudeDetector_YOLO:
    def __init__(self, checkpoint_path, device="cuda", inference_resolution=640, conf=0.):
        self.model = YOLO(checkpoint_path)
        self.device = device
        self.model.to(self.device)
        self.input_width = inference_resolution
        self.input_height = inference_resolution

        self.resize_transform2 = transforms.Resize((self.input_width, self.input_height))
        self.predictor = self.model._smart_load("predictor")(overrides={"conf": conf, "batch": 1, "save": False, "mode": "predict"})
        self.predictor.setup_model(model=self.model.model)

    def detect(self, img):
        img = self.resize_transform2(img).clamp(0, 1)
        output = self.model.model(img)
        preds = ops.non_max_suppression(output, self.predictor.args.conf, self.predictor.args.iou, self.predictor.args.classes,
                                        self.predictor.args.agnostic_nms,max_det=self.predictor.args.max_det,
                                        nc=len(self.model.names),end2end=getattr(self.predictor.args.model, "end2end", False),
                                        rotated=self.predictor.args.task == "obb",)
        total_scores = torch.zeros((len(preds),), device=img.device)
        total_masks = torch.zeros((len(preds),), device=img.device)
        for idx, pred in enumerate(preds):
            mask = torch.isin(pred[:, 5], torch.tensor([3, 4, 6, 14], device=img.device))
            total_scores[idx] += pred[mask, 4].sum()
            total_masks[idx] += mask.sum()
        return total_scores, total_masks

    def safety_score(self, img_list, target_label_list=None):
        
        batch_scores, batch_masks = self.detect(img_list)
        return batch_scores, batch_masks

