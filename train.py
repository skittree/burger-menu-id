import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os
from func_to_script import script
from torch.utils.data import Dataset
from yolov7 import create_yolov7_model
from yolov7.loss_factory import create_yolov7_loss
from yolov7.trainer import Yolov7Trainer, filter_eval_predictions
from yolov7.dataset import Yolov7Dataset, yolov7_collate_fn, create_yolov7_transforms
from yolov7.evaluation import CalculateMeanAveragePrecisionCallback
from pytorch_accelerated.schedulers import CosineLrScheduler
from pytorch_accelerated.callbacks import (
    EarlyStoppingCallback,
    SaveBestModelCallback,
    get_default_callbacks,
)
from functools import partial

class BurgerDatasetAdaptor(Dataset):
    def __init__(
        self,
        images_dir_path,
        annotations_dataframe,
        transforms=None,
    ):
        self.images_dir_path = images_dir_path
        self.annotations_df = annotations_dataframe
        self.transforms = transforms

        self.image_idx_to_image_id = {
            idx: image_id
            for idx, image_id in enumerate(self.annotations_df.image_id.unique())
        }
        self.image_id_to_image_idx = {
            v: k for k, v, in self.image_idx_to_image_id.items()
        }

    def __len__(self) -> int:
        return len(self.image_idx_to_image_id)

    def __getitem__(self, index):
        image_id = self.image_idx_to_image_id[index]
        image_info = self.annotations_df[self.annotations_df.image_id == image_id]
        file_name = image_info.image.values[0]
        assert image_id == image_info.image_id.values[0]

        image = Image.open(os.path.join(self.images_dir_path, file_name)).convert("RGB")
        image = np.array(image)

        image_hw = image.shape[:2]

        if image_info.has_annotation.any():
            xyxy_bboxes = image_info[["xmin", "ymin", "xmax", "ymax"]].values
            class_ids = image_info["class_id"].values
        else:
            xyxy_bboxes = np.array([])
            class_ids = np.array([])

        if self.transforms is not None:
            transformed = self.transforms(
                image=image, bboxes=xyxy_bboxes, labels=class_ids
            )
            image = transformed["image"]
            xyxy_bboxes = np.array(transformed["bboxes"])
            class_ids = np.array(transformed["labels"])

        return image, xyxy_bboxes, class_ids, image_id, image_hw

@script
def main(
    image_size: int = 640,
    pretrained: bool = True,
    num_epochs: int = 30,
    batch_size: int = 4,
):
    df = pd.read_csv("rico/burger.csv")
    train_ds = BurgerDatasetAdaptor("rico/combined/", df[df["split"]=="train"])
    val_ds = BurgerDatasetAdaptor("rico/combined/", df[df["split"]=="val"])
    yolo_train_ds = Yolov7Dataset(train_ds, create_yolov7_transforms((image_size, image_size), training=True))
    yolo_val_ds = Yolov7Dataset(val_ds, create_yolov7_transforms((image_size, image_size)))
    model = create_yolov7_model('yolov7', num_classes=1, pretrained=pretrained)
    criterion = create_yolov7_loss(model, image_size=image_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    trainer = Yolov7Trainer(
        model=model,
        optimizer=optimizer,
        loss_func=criterion,
        filter_eval_predictions_fn=partial(
            filter_eval_predictions, confidence_threshold=0.01, nms_threshold=0.3
        ),
        callbacks=[
            CalculateMeanAveragePrecisionCallback.create_from_targets_df(
                targets_df=df[df["split"]=="val"].query("has_annotation == True")[
                    ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
                ],
                image_ids=set(df[df["split"]=="val"].image_id.unique()),
                iou_threshold=0.2,
            ),
            SaveBestModelCallback(watch_metric="map", greater_is_better=True),
            EarlyStoppingCallback(
                early_stopping_patience=3,
                watch_metric="map",
                greater_is_better=True,
                early_stopping_threshold=0.001,
            ),
            *get_default_callbacks(progress_bar=True),
        ],
    )

    trainer.train(
        num_epochs=num_epochs,
        train_dataset=yolo_train_ds,
        eval_dataset=yolo_val_ds,
        per_device_batch_size=batch_size,
        create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(
            num_warmup_epochs=5,
            num_cooldown_epochs=5,
            k_decay=2,
        ),
        collate_fn=yolov7_collate_fn,
    )

if __name__ == "__main__":
    main()