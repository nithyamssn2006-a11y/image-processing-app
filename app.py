import os
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_enhance(img):
    enhanced = cv2.convertScaleAbs(img, alpha=1.3, beta=15)
    return enhanced


def apply_background_removal(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    h, w = img.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    result = img * mask2[:, :, np.newaxis]

    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    final = np.where(mask2[:, :, np.newaxis] == 1, result, white_bg)

    return final


def apply_morphology(img, operation):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)

    if operation == "erode":
        result = cv2.erode(binary, kernel, iterations=1)
    elif operation == "dilate":
        result = cv2.dilate(binary, kernel, iterations=1)
    elif operation == "open":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        result = binary

    return result


def apply_threshold(img, method):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "binary":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif method == "otsu":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        result = gray

    return result


@app.route("/", methods=["GET", "POST"])
def index():
    processed_img = None

    if request.method == "POST":
        file = request.files.get("file")
        operation = request.form.get("operation", "grayscale")

        if file and file.filename != "":
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            print("File saved at:", filepath)

            img = cv2.imread(filepath)

            if img is None:
                print("ERROR: Could not read image at", filepath)
                return render_template("index.html", processed_img=None)

            if operation == "grayscale":
                result = apply_grayscale(img)
            elif operation == "enhance":
                result = apply_enhance(img)
            elif operation == "background":
                result = apply_background_removal(img)
            elif operation in ["erode", "dilate", "open", "close"]:
                result = apply_morphology(img, operation)
            elif operation in ["binary", "otsu", "adaptive"]:
                result = apply_threshold(img, operation)
            elif operation == "blur":
                result = cv2.GaussianBlur(image, (15, 15), 0)
            else:
                result = img

            output_filename = operation + "_" + file.filename
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            cv2.imwrite(output_path, result)

            processed_img = output_path.replace("\\", "/")
            print("Processed image saved at:", processed_img)
        else:
            print("No file selected")

    return render_template("index.html", processed_img=processed_img)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)