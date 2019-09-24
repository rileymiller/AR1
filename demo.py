import cv2

def main():
    print("Hello")
    bgr_image = cv2.imread("roster.jpg")

    height = bgr_image.shape[0]
    width = bgr_image.shape[1]

    print("Size of this image: (width,height): (%d,%d)" % (width, height))

    # Show image.
    cv2.imshow("my image", bgr_image)

    # Wait for xx msec (0 means wait till a keypress).
    cv2.waitKey(0)


if __name__ == "__main__":
        main()
