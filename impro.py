import fitz
import numpy as np
import cv2 as cv
from io import BytesIO
from PIL import Image
import os


class utils:
    @staticmethod
    def rotate(img, angle, rotPoint=None):
        (h, w) = img.shape[:2]
        if rotPoint is None:
            rotPoint = (w // 2, h // 2)
        rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
        dims = (w, h)
        return cv.warpAffine(img, rotMat, dims, borderValue=255)

    @staticmethod
    def content_indexor(arr):
        arr_thresh = arr > 1
        arr_extended = np.insert(arr_thresh, [0, arr.shape[0]], 0)
        arr_shift_l = arr_extended[1:]
        arr_shift_r = arr_extended[:-1]

        start = np.argwhere((arr_shift_l > arr_shift_r) == True)
        end = np.argwhere((arr_shift_l < arr_shift_r) == True)

        cord_2 = np.concatenate((start, end), axis=1)
        cord_2 = np.delete(cord_2, np.argwhere((cord_2[:, 1] - cord_2[:, 0]) <= 3), axis=0)

        return cord_2


class process:
    @staticmethod
    def pdf2ImgArray(path: str, matrixSize: tuple = (4, 4)) -> np.array or str:
        try:
            mat = fitz.Matrix(matrixSize[0], matrixSize[1])
            pdf = fitz.open(path)
            page = pdf.load_page(0)
            img = page.get_pixmap(matrix=mat)
            img = Image.open(BytesIO(img.tobytes()))

            return np.array(img)
        except:
            return "ERR: s1:pdf2ImgArray"

    @staticmethod
    def edgeCutter(img: np.array, hcut: int = 5, vcut: int = 30) -> np.array or str:
        try:
            sh = img.shape
            return img[vcut:sh[0] - vcut, hcut:sh[1] - hcut]
        except:
            return "ERR: s2:edgeCutter"

    @staticmethod
    def threshold(img: np.array, threshold: int = 100) -> np.array or str:
        try:
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
            return img
        except:
            return "ERR: s3:threshold"

    @staticmethod
    def autoRotator(img: np.array, rot_portion: int = 5) -> np.array or str:
        try:
            portion = int(img.shape[0] / rot_portion)
            ub = np.argwhere(img[portion] == 0)[0][0]
            lb = np.argwhere(img[(rot_portion - 1) * portion] == 0)[0][0]
            hb = (rot_portion - 2) * portion
            deg = np.arctan((ub - lb) / hb) * 180 / np.pi
            return utils.rotate(img, deg)
        except:
            return "ERR: s4:autoRotator"

    @staticmethod
    def rescaleNegative(img: np.array, scale: float = 1 / 255) -> np.array or str:
        try:
            return 1 - img * scale
        except:
            return "ERR: s5:rescaleNegative"

    @staticmethod
    def borderCrop(img: np.array, PMrange: int = 200, hthresh: int = 120, vthresh: int = 200) -> np.array or str:
        try:
            portion_0 = int(img.shape[0] / 2)
            portion_1 = int(img.shape[1] / 2)
            mid_0 = np.argwhere(np.sum(img[portion_0 - PMrange:portion_0 + PMrange], axis=0) >= hthresh)
            mid_1 = np.argwhere(np.sum(img[:, portion_1 - PMrange:portion_1 + PMrange], axis=1) >= vthresh)
            return img[mid_1[0, 0]:mid_1[-1, 0], mid_0[0, 0]:mid_0[-1, 0]]
        except:
            return "ERR: s6:borderCrop"

    @staticmethod
    def resizer(img: np.array, size: tuple = (2520, 2000)):
        try:
            return cv.resize(img, size[::-1], interpolation=cv.INTER_AREA)
        except:
            return "ERR: s7:resizer"

    @staticmethod
    def findSubject(img: np.array, udCord: tuple=(254,330), lrCord: tuple=(1750,1905)) -> np.array or str:
        try:
            img = img[udCord[0]:udCord[1], lrCord[0]:lrCord[1]]

            sum_ = np.sum(img, axis=1)
            sum_ = sum_ > 0.1
            sum_sh = np.append(sum_[1:], False)

            for i in (np.argwhere(sum_ > sum_sh) + 1):
                if i > 10:
                    end = i[0]
                    break

            return img[:end]



        except:
            return "ERR: s8:findSubject"

    @staticmethod
    def segmentor(img: np.array) -> list or str:
        try:
            axis_x = np.sum(img, axis=0)
            axis_y = np.sum(img, axis=1)

            cord_x = utils.content_indexor(axis_x)
            cord_y = utils.content_indexor(axis_y)

            S_y = cord_y[0][0]
            E_y = cord_y[0][1]
            imgs = []
            for i, (S_x, E_x) in enumerate(cord_x):
                sub_img = img[S_y:E_y, S_x:E_x]
                imgs.append(1 - sub_img)
            return imgs
        except:
            return "ERR: s9:segmentor"
