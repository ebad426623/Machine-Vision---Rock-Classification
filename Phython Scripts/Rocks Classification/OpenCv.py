import sys
import ctypes
import numpy as np
import cv2
import csv

sys.path.append("../MvImport")
from MvCameraControl_class import *


# Global Variables
edge = 100
cv_image = None
detected_colors = [] 
csv_file = "rock_data.csv"


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

toEdge = int(input("Press 0 for Live Video and 1 for Live Edge Detetction: ")) 

# Get First Device
stDeviceList = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

# Create Camera Object
cam = MvCamera()
ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print(f"Failed to create handle! Error code: 0x{ret:X}")
    sys.exit()

# Open Device
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print(f"Failed to open device! Error code: 0x{ret:X}")
    cam.MV_CC_DestroyHandle()
    sys.exit()

# Set Camera Parameters
cam.MV_CC_SetFloatValue("ExposureTime", 15000.0)  # Set exposure time
cam.MV_CC_SetEnumValue("GainAuto", 0)  # Enable auto gain
cam.MV_CC_SetFloatValue("Gain", 0)  # Set Gain

# Start Grabbing
ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print(f"Failed to start grabbing! Error code: 0x{ret:X}")
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    sys.exit()

print("Camera is grabbing frames... Press ESC to exit.")


# Get Frame
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


# Edge Detection
def edge_detection(cv_image):
    global edge
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) 
    edges = cv2.Canny(blurred, edge, edge)
    edge -= 1

    if(edge < 30):
        edge = 100

    return edges


def detect_color(hue, sat, val):
    if sat < 50 and val > 200:
        return "White"
    
    elif val < 100:
        return "Black"
    
    elif (0 <= hue <= 10 or 170 <= hue <= 179):
        return "Red"
    
    elif 10 < hue <= 40:
        return "Yellow"
    
    elif 40 < hue <= 90:
        return "Green"
    
    elif 90 < hue <= 170:
        return "Blue"
    
    elif 160 < hue <= 170:
        return "Pink"



drawing = False
ix, iy = -1, -1 
fx, fy = -1, -1

# Mouse Callback Function
def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, cv_image, fx, fy, detected_colors
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down
        drawing = True
        ix, iy = x, y
        

    elif event == cv2.EVENT_LBUTTONUP:
        fx, fy = x, y
        if drawing:
            cv2.rectangle(cv_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
            drawing = False


            # Extract ROI
            roi = cv_image[min(iy, fy):max(iy, fy), min(ix, fx):max(ix, fx)]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_hsv = hsv_roi.mean(axis=(0, 1))
            avg_hsv_text = f"Avg HSV: ({avg_hsv[0]:.2f}, {avg_hsv[1]:.2f}, {avg_hsv[2]:.2f})"
            
            class_name = input("Enter Rock's Class [0, 1 ,2]: ")

            cv2.imshow("Camera Stream", cv_image)

            # Save HSV Values into CSV File
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([avg_hsv[0], avg_hsv[1], avg_hsv[2], class_name])

        
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv_image = getFrame()
        cv2.imshow("Camera Stream", cv_image)

    
    # elif event == cv2.EVENT_MBUTTONDOWN:
    #     hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
    #     hue = hsv_image[y, x, 0]
    #     sat = hsv_image[y, x, 1]
    #     val = hsv_image[y, x, 2]

    #     color_name = detect_color(hue, sat, val)
    #     print(f"Left button clicked at ({x}, {y}), HSV: ({hue}, {sat}, {val}), Detected Color: {color_name}")

    #     detected_colors.append((x, y, color_name))
    #     cv2.putText(cv_image, color_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
    #                 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    #     cv2.imshow("Camera Stream", cv_image)


# Main Loop
try:
    cv2.namedWindow("Camera Stream")
    cv_image = getFrame()
    cv2.setMouseCallback("Camera Stream", mouse_callback)
    while True:

        # cv_image = getFrame()
        if cv_image is None:
            continue

        if toEdge == 1:
            cv_image = edge_detection(cv_image)

        # for (x, y, color_name) in detected_colors:
        #     cv2.putText(cv_image, color_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
        #             1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display Image and Press ESC to exit
        cv2.imshow("Camera Stream", cv_image)


        if cv2.waitKey(0) == 27:
            break

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    # Stop Grabbing & Release Resources
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    print("Camera resources released.")
