import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization


SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0


class_ids = [
    "car",
    "pedestrian",
    "trafficLight",
    "biker",
    "truck",
    # no idea why the class index list did not contain the following classes which are in the dataset
    # will cause the next Python cell to error due to them missing :/
    # perhaps these were not included in the tutorial as they are newer additions to the dataset?
    "trafficLight-Red",
    "trafficLight-RedLeft",
    "trafficLight-Green",
    "trafficLight-GreenLeft",
    "trafficLight-Yellow",
    "trafficLight-YellowLeft"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

path_images = "./self-driving-car-images/"
path_annot = "./self-driving-car-annotations/"


xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)

# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
        if file_name.endswith(".jpg")
    ]
)


def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)


bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

# Determine the number of validation samples
num_val = int(len(xml_files) * SPLIT_RATIO)

# Split the dataset into train and validation sets
val_data = data.take(num_val)
train_data = data.skip(num_val)


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
        keras_cv.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
        ),
    ]
)

"""## Creating Training Dataset"""

train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.75, 1.3),
    bounding_box_format="xyxy",
)

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)



# prompt: use matplotlib to visualise dataset with bounding boxes 
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

def mpl_visualize_dataset_with_bboxes(dataset, bounding_box_format, value_range, rows, cols, dataset_split_name):
    """Visualizes a dataset with bounding boxes using matplotlib.

    Args:
        dataset: A tf.data.Dataset containing image and bounding box data.
        bounding_box_format: The format of the bounding boxes (e.g., 'xyxy').
        value_range: The range of pixel values in the images.
        rows: The number of rows in the visualization grid.
        cols: The number of columns in the visualization grid.
    """
    inputs = next(iter(dataset.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    axs = axs.flatten()

    for i in range(rows * cols):
        if i < len(images):
            image = images[i].numpy().astype(np.uint8)
            boxes = bounding_boxes["boxes"][i].numpy()
            classes = bounding_boxes["classes"][i].numpy().astype(int)

            axs[i].imshow(image)
            for box, cls in zip(boxes, classes):
                xmin, ymin, xmax, ymax = box
                axs[i].add_patch(
                    plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
                axs[i].text(
                    xmin,
                    ymin,
                    class_mapping[cls],
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )
            axs[i].axis("off")
        else:
            axs[i].axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.savefig(dataset_split_name+".png")
    plt.close()
    plt.show()


# Example usage:
mpl_visualize_dataset_with_bboxes(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2, dataset_split_name="train"
)
mpl_visualize_dataset_with_bboxes(
    val_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2, dataset_split_name="val"
)


def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"  # We will use yolov8 small backbone with coco weights
)

yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

"""
# ValueError: Invalid box loss for YOLOV8Detector: ciou. Box loss should be a keras.Loss or the string 'iou'
yolo.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)
"""

#yolo.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss=keras_cv.losses.CIoULoss())

# maybe investigate the difference between  Intersection over Union and Compute(d?) intersection over union https://github.com/keras-team/keras-cv/tree/master/keras_cv/src/losses
# if they ultimately just do the same thing except perhaps with varied results I am ok with that
yolo.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="iou")

class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs



yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3#,callbacks=[EvaluateCOCOMetricsCallback(val_ds, "model.h5")],
)
