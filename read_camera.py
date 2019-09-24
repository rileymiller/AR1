import cv2
import sys

def main():

    print('hello')
    # video_capture = cv2.VideoCapture(1)
    # video_capture = 'ad'
    print('after video capture')
    got_image, bgr_image_input = video_capture.read()

    if not got_image:
        print("can't read image input")
        sys.exit()

    while True:
        got_image, bgr_image_input = video_capture.read()
        cv2.imshow("Dank Meme", bgr_image_input)

        key_pressed = cv2.waitKey(1) & 0xFF



if __name__ == "__main__":
    main()
