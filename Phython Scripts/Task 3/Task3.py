import os
import sys
import cv2
import csv
import ctypes
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

sys.path.append("../MvImport")
from MvCameraControl_class import *


### GLOBAL VARIABLES
cam = None
file = "rock_data_from_input.csv"
cv_image = None
background = None
label_colors = {}
global coordsX, coordsY, pressedMouse
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors = 3) # using 3 neighbours


def getFrame():
    global cam
    # Initialize frame buffer
    stOutFrame = MV_FRAME_OUT()
    ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))

    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if ret != 0:
        print(f"Failed to get image buffer! Error code: 0x{ret:X}")
        return None  # Return None if there's an error

    # Convert to OpenCV Image
    buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
    ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)

    width, height = stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight
    scale_factor = max(1920 / width, 1080 / height)

    np_image = np.ctypeslib.as_array(buf_cache).reshape(height, width)
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_BayerBG2BGR)
    cv_image = cv2.resize(cv_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    cam.MV_CC_FreeImageBuffer(stOutFrame)  # Free buffer after use

    return cv_image

def open_camera():
    global cam, cv_image
    MvCamera().MV_CC_Initialize()

    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

    if ret != 0:
        print(f"Device enumeration failed! Error code: 0x{ret:X}")
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("No camera devices found.")
        sys.exit()

    print(f"Found {deviceList.nDeviceNum} device(s).")

    stDeviceList = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"Failed to create handle! Error code: 0x{ret:X}")
        sys.exit()

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"Failed to open device! Error code: 0x{ret:X}")
        cam.MV_CC_DestroyHandle()
        sys.exit()

    cam.MV_CC_SetFloatValue("ExposureTime", 15000.0)  # Set exposure time
    cam.MV_CC_SetEnumValue("GainAuto", 0)  # Enable auto gain
    cam.MV_CC_SetFloatValue("Gain", 0)  # Set Gain

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"Failed to start grabbing! Error code: 0x{ret:X}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        sys.exit()

    print("Camera is grabbing frames... Press ESC to exit.")

def mouse_callback(event, x, y, flags, param):
    global coordsX, coordsY, pressedMouse
    if event == cv2.EVENT_LBUTTONDOWN:
        coordsX, coordsY = x, y
        pressedMouse = True

def background_subtraction(frame):
    global background

    threshold_value = cv2.getTrackbarPos("Threshold", "Controls")
    kernel_size = cv2.getTrackbarPos("Kernel Size", "Controls")

    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    gray_background = cv2.cvtColor(background_resized, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fg_mask = cv2.absdiff(gray_background, gray_frame)

    _, thresh = cv2.threshold(fg_mask, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.namedWindow("After Thresholding", cv2.WINDOW_NORMAL)
    cv2.imshow("After Thresholding", thresh)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    cv2.namedWindow("After Morph", cv2.WINDOW_NORMAL)
    cv2.imshow("After Morph", clean_mask)

    return clean_mask

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def perform_task():
    global cv_image, cam
    try:
        # cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback("Original Image", mouse_callback)

        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Threshold", "Controls", 25, 255, lambda x: None)
        cv2.createTrackbar("Kernel Size", "Controls", 16, 100, lambda x: None)
        cv2.createTrackbar("Minimum Area", "Controls", 3000, 100000, lambda x: None)

        while True:
            if (cv_image := getFrame()) is None: continue
            # cv2.imshow("Original Image", cv_image)

            mask = background_subtraction(cv_image)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            min_area = cv2.getTrackbarPos("Minimum Area", "Controls")

            # Create a three-channel image for coloring the masks
            colored_image = np.zeros_like(cv_image)

            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    x, y, w, h, _ = stats[label]

                    # Classify the object
                    object_label = classify_object(cv_image, labels == label)

                    # Draw bounding box and label
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"Class {object_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if label not in label_colors:
                        label_colors[label] = get_random_color()

                    # Assign the color to the three-channel image
                    colored_image[labels == label] = label_colors[label]

            cv2.namedWindow("Final", cv2.WINDOW_NORMAL)
            cv2.imshow("Final", cv_image)
            cv2.namedWindow("Colored Mask", cv2.WINDOW_NORMAL)
            cv2.imshow("Colored Mask", colored_image)

            key = cv2.waitKey(1)

            if key == 27:
                break

            if key == ord('r'):
                capture_background()
                

    except KeyboardInterrupt:
        pass

    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("Camera is closed.")

def capture_background():
    global background
    print("Capturing background, please ensure no objects are in view...")
    background = getFrame()
    cv2.imwrite("images/background.png", background)
    print("Background captured.")

def train():
    global scaler, knn
    data = pd.read_csv(file, header=None)
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values  
    
    # normalizing values
    X = scaler.fit_transform(X)
    
    # training model
    knn.fit(X, y)

def classify_object(image, mask):
    """
    Classifies a detected object using the trained KNN model.
    """
    global scaler, knn

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = (mask * 255).astype(np.uint8)

    # compute color histograms for H, S, and V channels
    h_hist = cv2.calcHist([hsv_image], [0], mask, [180], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_image], [1], mask, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_image], [2], mask, [256], [0, 256]).flatten()

    # flatten histograms and normalize
    features = np.concatenate((h_hist, s_hist, v_hist)).reshape(1, -1)
    features = scaler.transform(features)

    # predict using KNN
    label = knn.predict(features)[0]

    return label

if __name__ == "__main__":
    open_camera()
    path = "images/background.png"
    if os.path.exists(path):
        background = cv2.imread(path)
    else:
        capture_background()
    train()
    perform_task()
