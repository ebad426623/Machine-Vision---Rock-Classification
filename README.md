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

In [OpenCv.py](./Python%20Scripts/Rocks%20Classification/OpenCv.py), I collected the dataset from the camera and saved it into [rock_data.csv](./Python%20Scripts/Rocks%20Classification/rock_data.csv). The stored features were the mean values of the HSV color space.

In [PredictRocks.py](./Python%20Scripts/Rocks%20Classification/PredictRocks.py), I used the previously collected data to train a KNN model, enabling real-time rock classification.

In [ClassificationUsingHistogram.py](./Python%20Scripts/Rocks%20Classification%20using%20Histogram/ClassificationUsingHistogram.py), I classified rocks using KNN by first converting the HSV color space into a histogram. I then used the histogram features to train the KNN model and performed real-time predictions. This method proved to be more accurate than using just the mean HSV values as features.

In [BackgroundSubtraction.py](/Python%20Scripts/Background%20Subtraction/BackgroundSubtraction.py), I use background subtraction, object detection, and classification to identify and categorize objects in a scene. First, I capture the background when no objects are present and store it as a reference image. During operation, I process each new frame by subtracting the background to isolate foreground objects, applying thresholding and morphological operations to refine the mask. I then use connected component analysis to detect individual objects, filtering out those below a minimum area threshold. To classify the detected objects, I extract their regions, convert them to the HSV color space, and compute color histograms as features. These features are normalized and passed to a pre-trained K-Nearest Neighbors (KNN) model for classification. By integrating these techniques, I efficiently detect and classify objects based on color features while minimizing noise.

## 6 Internship Acknowledgment

This project was developed during an internship at [Aremak Bilişim Teknolojileri](https://www.aremak.com.tr) under the supervision of Emrah Bala.
