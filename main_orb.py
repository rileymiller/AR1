import cv2
import sys

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
    kp, desc = detector.detectAndCompute(image=marker_image, mask=None)

    # Optionally draw keypoints on marker image.
    marker_image_display = cv2.cvtColor(src=marker_image, code=cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(image=marker_image, keypoints=kp, outImage=marker_image_display)
    cv2.imshow("Marker keypoints", marker_image_display)
    print("Detected %d keypoints on marker image" % len(kp))

    # Show image and wait for xx msec (0 = wait till keypress).
    key_pressed = cv2.waitKey(0)

if __name__ == "__main__":
    main()
