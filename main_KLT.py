import cv2
import sys
import time
import numpy as np

CAMERA_NUMBER = 0  # 0 is the default camera

def main():
    # Initialize image capture from camera.
    video_capture = cv2.VideoCapture(CAMERA_NUMBER)  # Open video capture object
    is_ok, bgr_image_input = video_capture.read()  # Make sure we can read video
    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    print("Hit any key to start tracking ...")
    while True:
        is_ok, bgr_image_input = video_capture.read()
        if not is_ok:
            break  # no camera, or reached end of video file

        # Show image and wait for xx msec (0 = wait till keypress).
        cv2.imshow("Input image", bgr_image_input)
        key_pressed = cv2.waitKey(1)
        if key_pressed != -1:
            break

    # Detect points to track in the input image.
    gray_image = cv2.cvtColor(src=bgr_image_input, code=cv2.COLOR_BGR2GRAY)
    image_pts = cv2.goodFeaturesToTrack(image=gray_image, maxCorners=100,
                            qualityLevel=0.01, minDistance=20, blockSize=11)
    image_pts = np.squeeze(image_pts)       # Get rid of singleton dimension, to get Nx2 array

    print("Hit any key to quit ...")
    while True:
        is_ok, bgr_image_input = video_capture.read()
        if not is_ok:
            break  # no camera, or reached end of video file

        new_gray = cv2.cvtColor(src=bgr_image_input, code=cv2.COLOR_BGR2GRAY)
        new_pts, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg=gray_image, nextImg=new_gray,
            prevPts=image_pts, nextPts=None,
            winSize=(32,32), maxLevel=8,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

        gray_image = new_gray.copy()
        image_pts = new_pts.copy()

        for p in image_pts:
            x = int(p[0])
            y = int(p[1])
            cv2.rectangle(bgr_image_input, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), 2)

        # Show image and wait for xx msec (0 = wait till keypress).
        cv2.imshow("Input image", bgr_image_input)
        key_pressed = cv2.waitKey(1)
        if key_pressed != -1:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
