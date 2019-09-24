import cv2
import sys

def main():

    # Read images from a video file in the current folder.
    video_capture = cv2.VideoCapture("sample_movie.avi")
    # Open video capture object
    got_image, bgr_image = video_capture.read()
    # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()
    # Read and show images until end of video is reached
    while True:
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break # End of video; exit the while loop

        cv2.imshow("my image", bgr_image)
        # Wait for xx msec (0 means wait till a keypress).
        cv2.waitKey(30)

    print("number of frames: ", video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


if __name__ == "__main__":
    main()