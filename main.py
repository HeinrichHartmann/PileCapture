#!/usr/bin/env python

import cv2
import logging
import os
import sys
from datetime import datetime
import click

import pandas as pd
import numpy as np
from time import sleep

import gphoto2 as gp

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


class CanonCam:
    def __init__(self):
        gp.check_result(gp.use_python_logging())
        camera = gp.check_result(gp.gp_camera_new())
        gp.check_result(gp.gp_camera_init(camera))
        L.info(gp.check_result(gp.gp_camera_get_summary(camera)))
        self.camera = camera

    def __del__(self):
        self.camera.exit()

    def capture(self):
        L.info("Canon capture")
        file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
        target = os.path.join("/tmp", f"{file_path.name}")
        self.camera.file_get(
            file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
        ).save(target)
        return cv2.imread(target)


class WebCam:
    @classmethod
    def list(cls):
        "Returns a list of valid capture devices"
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
                cap.release()
            index += 1
        return arr

    def __init__(self, i=0):
        self.camera = cv2.VideoCapture(i)

    def __del__(self):
        self.camera.release()

    def capture(self):
        L.info("Webcam capture")
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("failed to grab frame")
        return frame


class DirCam:
    def __init__(self, pat):
        import glob

        pat = pat or "*.png"
        self.i = glob.iglob(pat)

    def capture(self):
        p = next(self.i)
        print(p)
        return cv2.imread(p)


def img_rot(img, n):
    "rotate image n-times 90-degrees"
    for i in range(n % 4):
        img = cv2.transpose(img)
        img = cv2.flip(img, flipCode=1)
    return img


def box2rect(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew


# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def rectify(image, box):
    rect = box2rect(box)
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
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def docdetect(i):
    o = i.copy()
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    i = cv2.GaussianBlur(i, (17, 17), 0)
    i = cv2.Canny(i, 0, 30)
    e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    i = cv2.morphologyEx(i, cv2.MORPH_CLOSE, e)
    (contours, _) = cv2.findContours(i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        cv2.drawContours(o, [c], -1, (0, 255, 0), 2)
        c = cv2.convexHull(c)
        cv2.drawContours(o, [c], -1, (255, 0, 0), 2)
        box = cv2.boxPoints(cv2.minAreaRect(c))
        box = np.int0(box)
        cv2.drawContours(o, [box], 0, (0, 0, 255), 2)
        return o, box
    raise Exception("No box found")


@click.command()
@click.option("--device_type", default="VideoCapture")
@click.option("--device_arg", default=None)
@click.option("--delay", default=1000)
def cli(device_type, device_arg, delay):
    if device_type == "VideoCapture":
        dev = WebCam
    elif device_type == "Canon":
        dev = CanonCam
    elif device_type == "Dir":
        dev = DirCam

    cam = dev(device_arg)
    rot = 0
    crop = False
    delay_swap = 0
    while True:
        img = cam.capture()
        img = img_rot(img, rot)
        img_orig = img.copy()
        img, box = docdetect(img)
        if crop:
            cv2.imshow("preview", rectify(img, box))
        else:
            cv2.imshow("preview", img)
        cv2.setWindowProperty("preview", cv2.WND_PROP_TOPMOST, 1)

        k = cv2.waitKey(delay) % 256
        if k in {27, ord("q")}:
            break
        elif k == ord("r"):  # right rotation
            rot += 1
        elif k == ord("l"):  # left rotation
            rot -= 1
        elif k == ord("c"):  # crop
            crop = not crop
        elif k == ord("w"):  # wait for keybpress
            delay, delay_swap = delay_swap, delay
        elif k == ord("s"):
            file_name_orig = (
                "data/"
                + datetime.now().isoformat().replace("T", " Image-")
                + ".orig.png"
            )
            file_name_rect = (
                "data/"
                + datetime.now().isoformat().replace("T", " Image-")
                + ".rect.png"
            )
            img_rect = rectify(img, box)
            cv2.imwrite(file_name_orig, img_orig)
            cv2.imwrite(file_name_rect, img_rect)
            L.info(f"{file_name_rect} written")


if __name__ == "__main__":
    cli()
