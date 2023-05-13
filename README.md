# Burger Menu Identifier using YOLOv7

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cdQV11JES0zJHGQ2ERi-8HyeYXbsgsqx?usp=sharing)

This repository contains the full process of creating a fine-tuned YOLOv7 model in PyTorch that identifies hamburger menus in mobile user interfaces. The model was trained on the RICO dataset and uses the [Yolov7-training](https://github.com/Chris-hughes10/Yolov7-training.git) repository as a base.

## Installation Instructions

To install and use this repository, follow these steps:

1. Install [Yolov7-training](https://github.com/Chris-hughes10/Yolov7-training.git) by running the following command:

```bash
   pip install git+https://github.com/Chris-hughes10/Yolov7-training.git
```

2. Install [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) and then PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

3. Install the rest of the required packages by running the following command:

```bash
   pip install -r requirements.txt
```

4. Save the pre-trained model [burger.pt](https://drive.google.com/file/d/1H33oeDgvestog0yPBioJUhdOvfrNnfuK/view?usp=sharing) and place it in the working directory.

## Usage

The repository contains the following notebooks:

- `creating_labels.ipynb`: for creating a hamburger menu dataset from the Rico dataset
- `training_model.ipynb`: for the main training loop
- `prediction.ipynb`: for automatic detection of hamburger menus in images

To use the model, run the `prediction.ipynb` notebook. The notebook automatically reads every JPG in the working directory and labels them with red bounding boxes.

You can adjust the input parameters to test the model's performance on different inputs. You can also view the model's output for each input by clicking on the "Run" button.