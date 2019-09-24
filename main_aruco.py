import sys
import cv2

CAMERA_NUMBER = 0       # 0 is the default camera

def main():
    # Initialize image capture from camera.
    video_capture = cv2.VideoCapture(CAMERA_NUMBER)     # Open video capture object
    is_ok, bgr_image_input = video_capture.read()       # Make sure we can read video
    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    # Get the pattern dictionary for 4x4 markers, with ids 0 through 99.
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    # Optionally show all markers in the dictionary.
    for id in range(0, 100):
        img = cv2.aruco.drawMarker(dictionary=arucoDict, id=id, sidePixels=200)
        cv2.imshow("img", img)
        cv2.waitKey(30)

    while True:
        is_ok, bgr_image_input = video_capture.read()
        if not is_ok:
            break               # no camera, or reached end of video file

        # Detect a marker.  Returns:
        #   corners:	list of detected marker corners; for each marker, corners are clockwise)
        #   ids: 	vector of ids for the detected markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=bgr_image_input,
            dictionary=arucoDict
        )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(
                image=bgr_image_input, corners=corners, ids=ids, borderColor=(0, 0, 255))

        cv2.imshow("Input image", bgr_image_input)

        # Show images and wait for xx msec (0 = wait till keypress).
        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed == 27 or key_pressed == ord('q'):
            break       # Quit on ESC or q

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

