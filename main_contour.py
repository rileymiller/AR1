import cv2
import sys

def main():
    # Read images from a video file in the current folder.
    video_capture = cv2.VideoCapture("square.wmv")     # Open video capture object
    got_image, bgr_image = video_capture.read()       # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()

    while True:
        got_image, bgr_image = video_capture.read()
        if not got_image:
            break       # End of video; exit the while loop

        # Convert color image to gray image.
        gray_image = cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2GRAY)   # convert BGR to grayscale
        output_thresh, binary_image = cv2.threshold(
            src=gray_image, maxval=255,
            type=cv2.THRESH_OTSU,  # determine threshold automatically from image
            thresh=0)  # ignore this if using THRESH_OTSU
        cv2.imshow("binary image", binary_image)

        # Find all contours in the image and draw them.
        contours, hierarchy = cv2.findContours(
            image=binary_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=bgr_image, contours=contours, color=(0,0,255), thickness=2,
                         contourIdx=-1)     # -1 means draw all contours

        cv2.imshow("my image", bgr_image)

        cv2.waitKey(100)        # Wait for xx msec (0 means wait till a keypress)

if __name__ == "__main__":
    main()
