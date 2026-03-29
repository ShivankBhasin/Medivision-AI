import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    def __init__(self):
        self.debug_mode = False

    def preprocess(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, bytes):
            img = cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_input, Image.Image):
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            img = image_input

        if img is None:
            raise ValueError("Could not load image")
            
        original = img.copy()

        img = self._resize_image(img)
        img = self._denoise(img)
        img = self._correct_skew(img)
        img = self._enhance_contrast(img)
        img = self._binarize(img)

        return img, original
        
    def _resize_image(self, img, max_width=2000):
        height, width = img.shape[:2]
        
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img
        
    def _denoise(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
    def _correct_skew(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:10]:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                angles.append(angle)

            median_angle = np.median(angles)

            if abs(median_angle) > 0.5 and abs(median_angle) < 45:
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (width, height),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
        return img
        
    def _enhance_contrast(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _binarize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

        return binary
