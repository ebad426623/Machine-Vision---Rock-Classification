# For Training KNN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# For Using Camera for Rock Detection
import cv2
import sys
import ctypes
import warnings
import numpy as np

sys.path.append("../MvImport")
from MvCameraControl_class import *

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Training KNN Model
file_path = 'rock_data.csv'
df = pd.read_csv(file_path, header=None, names=['Hue', 'Saturation', 'Brightness', 'Label'])

X = df[['Hue', 'Saturation', 'Brightness']]
y = df['Label']

scaler = StandardScaler()
X = scaler.fit_transform(X)


k = 3 
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)


# # Machine Vision Camera
# Global Variables
cv_image = None
box_x1, box_y1, box_x2, box_y2 = 750, 650, 900, 800 

# Initialize Camera SDK
MvCamera().MV_CC_Initialize()

# Enumerate Devices
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


def getFrame():
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
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = cv_image[box_y1:box_y2, box_x1:box_x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv_roi, axis=(0, 1))
        avg_hsv = scaler.transform([avg_hsv])
        pred = knn.predict(avg_hsv)[0]
        print(f"Predicted Rock: {pred + 1}")


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