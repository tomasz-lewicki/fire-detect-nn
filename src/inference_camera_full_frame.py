import torch
import os
import time
import cv2
import numpy as np

from model import Model

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=224,
    display_height=224,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def avgerage(l):
    return sum(l)/len(l)

# We can get some nice 8FPS from this image

if __name__ == "__main__":
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    device = torch.device("cuda:0")

    model = torch.load('weights/resnet50-acc=0.98.pt')
    model = model.to(device)

    if cap.isOpened():
        while True:
            start = time.time()
            ret_val, img = cap.read()
            img = np.swapaxes(img,0,2) # WxHxchannel convention to channelxWxH convention
            img_mini_batch = np.expand_dims(img, axis=0)
            tens = torch.Tensor(img_mini_batch).to(device)
            result = model(tens)
            print(result)
            print(time.time()-start)
            
        cap.release()

    else:
        print("Unable to open camera")


