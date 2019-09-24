import sys
import cv2
import numpy as np

CAMERA_NUMBER = 0       # 0 is the default camera
MARKER_LENGTH = 2.0
CUBE_LENGTH = MARKER_LENGTH

def main():
    # Initialize image capture from camera.
    video_capture = cv2.VideoCapture(CAMERA_NUMBER)     # Open video capture object
    is_ok, bgr_image_input = video_capture.read()       # Make sure we can read video
    if not is_ok:
        print("Cannot read video source")
        sys.exit()
    image_height = bgr_image_input.shape[0]
    image_width = bgr_image_input.shape[1]

    # Define camera intrinsics (ideally you should do a calibration to estimate these).
    f = 0.8 * bgr_image_input.shape[1]  # Guess focal length in pixels
    cx = image_width / 2  # Assume principal point is in center of image
    cy = image_height / 2
    dist_coeff = np.zeros(4)  # Assume no lens distortion
    K = np.float64([[f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1.0]])

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

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

            rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners, markerLength=MARKER_LENGTH,
                cameraMatrix=K, distCoeffs=dist_coeff
            )

            # Get the pose of the first detected marker with respect to the camera.
            rvec_m_c = rvecs[0]                 # This is a 1x3 rotation vector
            tm_c = tvecs[0]                     # This is a 1x3 translation vector

            # Draw coordinate axes for the marker.
            cv2.aruco.drawAxis(
                image=bgr_image_input, cameraMatrix=K, distCoeffs=dist_coeff,
                rvec=rvec_m_c, tvec=tm_c, length=MARKER_LENGTH)

        cv2.imshow("Input image", bgr_image_input)

        # Show images and wait for xx msec (0 = wait till keypress).
        key_pressed = cv2.waitKey(0) & 0xFF

        if key_pressed == 27 or key_pressed == ord('q'):
            break       # Quit on ESC or q

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

