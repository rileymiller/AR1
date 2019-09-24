import cv2
import sys
import time
import numpy as np

CAMERA_NUMBER = 0  # 0 is the default camera
MIN_MATCHES_FOR_DETECTION = 10    # Minimum number of matches for detection; increase for fewer false detections

def main():
    # Read in an image, and convert it to grayscale.
    marker_image = cv2.imread("stones.jpg", cv2.IMREAD_GRAYSCALE)
    if marker_image is None:
        print("Can't read marker image from file")
        sys.exit()
    # Optionally reduce the size of the marker image.
    h = int(0.5 * marker_image.shape[0])
    w = int(0.5 * marker_image.shape[1])
    marker_image = cv2.resize(src=marker_image, dsize=(w, h))

    # Initialize feature detector.
    detector = cv2.ORB_create(nfeatures=2500,  # default = 500
                              edgeThreshold=16)  # default = 31

    # Detect keypoints in marker image and compute descriptors.
    kp1, desc1 = detector.detectAndCompute(image=marker_image, mask=None)

    # Initialize image capture from camera.
    video_capture = cv2.VideoCapture(CAMERA_NUMBER)  # Open video capture object
    is_ok, bgr_image_input = video_capture.read()  # Make sure we can read video
    if not is_ok:
        print("Cannot read video source")
        sys.exit()

    while True:
        is_ok, bgr_image_input = video_capture.read()
        if not is_ok:
            break  # no camera, or reached end of video file

        # Set this to true if at any point in the processing we can't find the marker.
        unable_to_find_marker = False

        # Detect keypoints and compute descriptors for input image.
        image_input_gray = cv2.cvtColor(src=bgr_image_input, code=cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(image=image_input_gray, mask=None)
        if len(kp2) < MIN_MATCHES_FOR_DETECTION:  # Higher threshold - fewer false detections
            unable_to_find_marker = True

        # Match descriptors to marker image.
        if not unable_to_find_marker:
            matcher = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=False)
            matches = matcher.knnMatch(desc1, desc2, k=2)  # Find closest 2
            good_matches = []
            for m in matches:
                if m[0].distance < 0.8 * m[1].distance:  # Ratio test
                    good_matches.append(m[0])
            if len(good_matches) < MIN_MATCHES_FOR_DETECTION:  # Higher threshold - fewer false detections
                unable_to_find_marker = True

        # Fit homography.
        if not unable_to_find_marker:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            Hmat, mask = cv2.findHomography(
                srcPoints=src_pts, dstPoints=dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=5.0,  # default is 3.0
                maxIters=2000  # default is 2000
            )
            num_inliers = sum(mask)  # mask[i] is 1 if point i is an inlier, else 0
            if num_inliers < MIN_MATCHES_FOR_DETECTION:
                unable_to_find_marker = True

        # Draw marker border on the image.
        if not unable_to_find_marker:
            # Project the marker border lines to the image using the computed homography.
            h, w = marker_image.shape
            marker_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
            warped_corners = cv2.perspectiveTransform(marker_corners.reshape(-1, 1, 2), Hmat)

            cv2.polylines(img=bgr_image_input, pts=[np.int32(warped_corners)], isClosed=True,
                          color=[0, 255, 0], thickness=4, lineType=cv2.LINE_AA)

        # Show image and wait for xx msec (0 = wait till keypress).
        cv2.imshow("Input image", bgr_image_input)
        key_pressed = cv2.waitKey(0) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break  # Quit on ESC or q

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
