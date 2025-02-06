# Rock Classification Using OpenCV Object Detection

## 1 Project Overview

This project involves classifying three different classes of rocks using OpenCV object detection. The key steps in the project include:

- Capturing a dataset using a Hikrobot camera and lighting.
- Converting images to the HSV feature space.
- Storing the extracted features in a CSV file.
- Training a K-Nearest Neighbors (KNN) model on the collected data.
- Performing real-time rock classification using live video feeds.

The model successfully classified most of the rocks with high accuracy.

## 2 Equipment and Technologies Used

- **Camera Model:** Hikrobot MV-CS060-10UC-PRO
- **Lens:** MVL-HF0828M-6MPE
- **Camera Stand:** Aremak Adjustable Machine Vision Test Stand
- **Lighting:** Hikrobot Flat Light (MV-LBES-120-120-Y35-W)
- **Lighting Unit:** Hikrobot Light Control Units
- **Operating System:** Windows
- **Software Tools:** Python, OpenCV, Hikrobot SDK, NumPy

## 3 Setup Photos

![Setup Image](images/my-setup.jpg)
![Setup Image](images/my-setup1.jpg)
![Setup Image](images/my-setup2.jpg)

## 4 Installation and Running Instructions 

### Installation

Run the following command to install the necessary dependencies:

```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install opencv-python
```
Run the following command to collect data samples inside "Rocks Classification" folder:

```bash
python OpenCV.py
```

Run the following command to classify rocks in real time:

```bash
python PredictRocks.py
```
## 5 Code Documentation 

## 6 Internship Acknowledgment 

This project was developed during an internship at [Aremak Bilişim Teknolojileri](https://www.aremak.com.tr) under the supervision of Emrah Bala.

