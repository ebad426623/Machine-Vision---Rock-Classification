import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt



####################################################################################
########################### Phase 1, Data Collection ###############################
####################################################################################

# Global Variables
csv_file = "rock_data1.csv"

def plot_histogram(h, s, v, name, loc):
    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(h, color='r')
    plt.title('Hue')
    plt.subplot(132)
    plt.plot(s, color='g')
    plt.title('Saturation')
    plt.subplot(133)
    plt.plot(v, color='b')
    plt.title('Value')
    
    # Save the figure
    plt.savefig(f"{loc}/{name}.png")
    plt.close()

def create_csv():
    global h1, s1, v1, h2, s2, v2, h3, s3, v3
    class_folder = [[1, 14], [2, 20], [3, 13]]

    for i in range(3):
        for j in range(class_folder[i][1]):
            image = cv2.imread(f"../../dataset/class_{i+1}/C{i+1}_{j+1}.jpg")
            hsv_colors = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Histograms
            hist_hue = cv2.calcHist([hsv_colors], [0], None, [180], [0, 180])
            hist_sat = cv2.calcHist([hsv_colors], [1], None, [256], [0, 256])
            hist_val = cv2.calcHist([hsv_colors], [2], None, [256], [0, 256])

            name = f"C{i+1}_{j+1}_Histogram"
            loc = f"../../dataset/class_{i+1}"

            plot_histogram(hist_hue, hist_sat, hist_val, name, loc)

            hist_hue_flat = hist_hue.flatten()
            hist_sat_flat = hist_sat.flatten()
            hist_val_flat = hist_val.flatten()

            hist_combined = np.concatenate([hist_hue_flat, hist_sat_flat, hist_val_flat])
            hist_combined = np.append(hist_combined, i+1)

            with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(hist_combined)


####################################################################################
########################### Phase 2, Model Train ###################################
####################################################################################     
import os   
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

# Global Variables
knn = None
scaler = None

def train():
    global knn, scaler

    file_path = csv_file
    df = pd.read_csv(file_path, header=None)

    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k = 10
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = knn.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')


    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # k = 3
    # knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(X, y)

    # ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] 
    # for k in ks:
    #     knn = KNeighborsClassifier(n_neighbors=k)
        
    #     # Train the model
    #     knn.fit(X_train, y_train)
        
    #     # Evaluate the model
    #     accuracy = knn.score(X_test, y_test)
    #     print(f"K = {k}, Model accuracy: {accuracy:.2f}")


####################################################################################
########################### Phase 3, Rock Classification ###########################
####################################################################################
import sys
sys.path.append("../MvImport")
from MvCameraControl_class import *

# Global Variables
cv_image = None
cam = None
box_x1, box_y1, box_x2, box_y2 = 750, 650, 900, 800 


def getFrame():
    global cam
    # Initialize frame buffer
    stOutFrame = MV_FRAME_OUT()
    ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))

    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if ret != 0:
        print(f"Failed to get image buffer! Error code: 0x{ret:X}")
        return  # Skip this frame if there's an error

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

def mouse_callback(event, x, y, flags, param):
    global box_x1, box_x2, box_y1, box_y2, scaler
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = cv_image[box_y1:box_y2, box_x1:box_x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        
        hist_hue = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        hist_sat = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
        hist_val = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])


        hist_hue_flat = hist_hue.flatten()
        hist_sat_flat = hist_sat.flatten()
        hist_val_flat = hist_val.flatten()

        hist_combined = np.concatenate([hist_hue_flat, hist_sat_flat, hist_val_flat]).reshape(1, -1)

        hist_combined = scaler.transform(hist_combined)

        pred = knn.predict(hist_combined)[0]
        print(f"Predicted Rock: {pred}")


def classify_rock():
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

    

    try:
        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", mouse_callback)
        while True:
            cv_image = getFrame()
            if cv_image is None:
                continue

            
            cv2.rectangle(cv_image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2) 
            cv2.putText(cv_image, "Place the rock here", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
            cv2.imshow("Camera", cv_image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass

    finally:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("Camera is closed.")


    pass

if __name__ == "__main__":
    # print("Data Collection Started...")
    # create_csv()
    # print("Data Collection Completed...")
    # print()

    print("Model Training Started...")
    train()
    print("Model Training Completed...")
    print()

    print("Live Rock Classification...")
    classify_rock()
    print("Live Rock Classification Completed...")

    






















