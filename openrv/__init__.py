from openrv.app import *
from openrv.colors import *
from openrv.contour import *
from openrv.image import *
import openrv.utils

import cv2

BINARY     = cv2.THRESH_BINARY
BINARY_INV = cv2.THRESH_BINARY_INV

OTSU = cv2.THRESH_OTSU

TOZERO     = cv2.THRESH_TOZERO
TOZERO_INV = cv2.THRESH_TOZERO_INV

TRUNC = cv2.THRESH_TRUNC
TRIANGLE = cv2.THRESH_TRIANGLE