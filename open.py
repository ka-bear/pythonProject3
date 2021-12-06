import time

import cv2
import numpy as np

if __name__ == '__main__':
    '''
    lst = [r"C:\\Users\\User\\Downloads\\untitledd_LH0MdSUJ.mp4"]
    # The lower and upper bound for colour filtering
    colours = [np.array([0, 0, 0]), np.array([85, 91, 114])]

    # FPS of video
    fps = 30

    for video in lst:
        vid = cv2.VideoCapture(video)
        t = 0
        data = np.array([[0, 0, 0]])
        while True:
            jed, frame = vid.read()
            if not jed: break

            frame = frame[0:0+1000, 0:150+1300]

            # It converts the BGR colour space to RGB colour space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preparing the mask produced by the colour filter
            masks = []
            mask = cv2.inRange(rgb, colours[0], colours[1])
            for i in range(0, len(colours), 2):
                masks.append(cv2.inRange(rgb, colours[i], colours[i+1]))
                mask = cv2.bitwise_or(mask, masks[-1])

            # Blur to reduce noise
            img = cv2.GaussianBlur(mask, (5, 5), 0)
            img = cv2.medianBlur(img, 11)

            # More noise reduction
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)
            img = cv2.dilate(img, kernel, iterations=1)

            # Actual circle detection
            # param2 -> Higher = Higher circle standards
            #        -> Lower = Lower circle standards
            # minRadius -> Minimum radius
            # maxRadius -> Maximum radius
            # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html
            #circles = cv2.ellipse(img)
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=15, minRadius=300, maxRadius=600)

            if circles is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for circle in circles[0]:
                    # Draw circles
                    if ((data[-1][1] - circle[0]) ** 2 + (data[-1][2] - circle[1]) ** 2) ** 0.5 / (t - data[-1][0]) \
                             / 1000 > 50 and t != 0:
                        break

                    data = np.append(data, np.array([[t, circle[0], circle[1]]]), axis=0)
                    img = cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 0, 255), 5)
                    break

            # Displaying the result
            cv2.imshow('result', img)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

            t += 1 / fps
            print(round(t, 3))

        # Writing the data
        name = video.split("\\")[-1].replace(".MP4", "")
        with open(f"C:/Users/jedli/Music/{name}.csv", "w+") as f:
            for j in range(1, len(data)):
                f.write(f"{data[j][0]},{data[j][1]},{data[j][2]}\n")

        # Finding the circle centre
        # (x - x0) ^ 2 + (y - y0) ^ 2 = r ^ 2
    '''

    lst = [r"C:\Users\User\Downloads\MAH03974.MP4"]

    # The lower and upper bound for colour filtering
    colours = [np.array([0, 0, 0]), np.array([85, 91, 114])]

    # FPS of video
    fps = 100

    for video in lst:
        vid = cv2.VideoCapture(video)
        t = 0
        data = np.array([[0, 0, 0]])
        while vid.isOpened():
            jed, frame = vid.read()
            if not jed: break

            frame = frame[500:900, 350:800]

            # It converts the BGR colour space to RGB colour space
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preparing the mask produced by the colour filter
            masks = []
            mask = cv2.inRange(rgb, colours[0], colours[1])
            for i in range(0, len(colours), 2):
                masks.append(cv2.inRange(rgb, colours[i], colours[i + 1]))
                mask = cv2.bitwise_or(mask, masks[-1])

            # Blur to reduce noise
            img = cv2.GaussianBlur(mask, (5, 5), 0)

            #
            #
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


            #(xc, yc) = np.array(img.shape) / 2

            #dxy = [np.sqrt((yc - cnt.squeeze().mean(0)[1]) ** 2 + (xc - cnt.squeeze().mean(0)[0]) ** 2) for cnt in contours]

            best = []  # this is the contour closest to the center of the image
            for i in contours:
                if len(best)<len(i):
                    best = i
            print()
            img = cv2.bitwise_not(img)
            #cv2.drawContours(img, best, -1, 156 , 3)
            #pupil = contours[best]
            (xc, yc), (a, b), theta = cv2.fitEllipse(best)
            print(cv2.fitEllipse(best))
            #cv2.drawContours(img,cv2.fitEllipse(best),-1,255,3)
            image = cv2.ellipse(img, cv2.fitEllipse(best), 155, 3)
            cv2.imshow("a", img)
            if (cv2.waitKey(1) & 0xFF == ord("q")):
                break
            # (xc, yc) = np.array(gray.shape) / 2
            #
            # dxy = [np.sqrt((yc - cnt.squeeze().mean(0)[1]) ** 2 + (xc - cnt.squeeze().mean(0)[0]) ** 2) for cnt in contours]
            # best = np.argsort(dxy)[0]  # this is the contour closest to the center of the image
            # pupil = contours[best]
            # (xc, yc), (a, b), theta = cv2.fitEllipse(pupil)
