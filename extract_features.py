import cv2
import numpy as np

def extract_features(image_path):
    """
    Extracts brightness, contrast, saturation, and sharpness features from an image.
    Returns valid values even if processing fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("❌ Failed to read image. Ensure it's a valid image file.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        return {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "saturation": round(saturation, 2),
            "sharpness": round(sharpness, 2)
        }

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return {
            "brightness": 50,
            "contrast": 20,
            "saturation": 30,
            "sharpness": 10
        }
