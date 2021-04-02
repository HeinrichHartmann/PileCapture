import cv2
import numpy as np
from time import sleep

import pandas as pd
import cv2


ROT=0

def list_cap():
    url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
    table = pd.read_html(url)[0]
    table.columns = table.columns.droplevel()
    cap = cv2.VideoCapture(1)
    resolutions = {}
    for index, row in table[["W", "H"]].iterrows():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resolutions[str(width)+"x"+str(height)] = "OK"
    print(resolutions)

def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew

# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


class Scanner(object):

    # https://github.com/vipul-sharma20/document-scanner
    def detect_edge(self, image, enabled_transform = False):
        dst = None
        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edged = cv2.Canny(blurred, 0, 30)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        i = 0
        for cnt in contours:
            i += 1
            if i > 3: break
            epsilon = 0.051 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                target = approx
                cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
                approx = rectify(target)
                # dst = self.four_point_transform(orig, approx)
        return image, dst

def de2(i):
    o = i.copy()
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    i = cv2.GaussianBlur(i, (17, 17), 0)
    i = cv2.Canny(i, 0, 30)
    e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    i = cv2.morphologyEx(i, cv2.MORPH_CLOSE, e)
    (contours, _) = cv2.findContours(i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # o = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
    for c in contours:
        cv2.drawContours(o, [c], -1, (0, 255, 0), 2)
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)
        cv2.drawContours(o,[box],0,(0,0,255),2)
        return o, box
        # return four_point_transform(o, rectify(box)), box


def cam():
    global ROT
    f = 0
    cam = cv2.VideoCapture(0)
    scanner = Scanner()
    cv2.namedWindow("test")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        for i in range(ROT):
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame,flipCode=1)
        if not ret:
            print("failed to grab frame")
            break
        # image, dst = scanner.detect_edge(frame, True)
        # cv2.imshow("test", image)
        preview, box = de2(frame)
        cv2.imshow("test", preview)
        image = frame
        k = cv2.waitKey(33)
        if k%256 in { 27, ord('q') }:
            print("Closing")
            break
        elif k%256 == ord('s'):
            img_name = "opencv_frame_{}.png".format(img_counter)
            rect = rectify(box)
            out = four_point_transform(frame, rect)
            cv2.imwrite(img_name, out)
            print("{} written!".format(img_name))
            cv2.imshow("test", out)
            cv2.waitKey(2000)
            img_counter += 1
        elif k%256 == ord('r'):
            ROT += 1
        elif k%256 == ord('l'):
            ROT -= 1
    cam.release()

    cv2.destroyAllWindows()


def img():
    img = cv2.imread("./test.png")
    cv2.imwrite("./out.png", de2(img))

cam()
# img()
