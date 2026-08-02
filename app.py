import os
import cv2
import numpy as np
import urllib.request
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ---------- Colorization model files ----------
PROTOTXT_PATH = os.path.join(MODEL_FOLDER, "colorization_deploy_v2.prototxt")
POINTS_PATH = os.path.join(MODEL_FOLDER, "pts_in_hull.npy")
MODEL_PATH = os.path.join(MODEL_FOLDER, "colorization_release_v2.caffemodel")

PROTOTXT_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt"
POINTS_URL = "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"
MODEL_URL = "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel"

_colorizer_net = None


def download_if_missing(url, path):
    if not os.path.exists(path):
        print("Downloading:", path)
        urllib.request.urlretrieve(url, path)
        print("Downloaded:", path)


def get_colorizer():
    global _colorizer_net
    if _colorizer_net is None:
        download_if_missing(PROTOTXT_URL, PROTOTXT_PATH)
        download_if_missing(POINTS_URL, POINTS_PATH)
        download_if_missing(MODEL_URL, MODEL_PATH)

        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        pts = np.load(POINTS_PATH)

        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype(np.float32)]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        _colorizer_net = net
    return _colorizer_net


def apply_colorization(img):
    net = get_colorizer()

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized


def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_enhance(img):
    enhanced = cv2.convertScaleAbs(img, alpha=1.3, beta=15)
    return enhanced


def apply_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    enhanced_lab = cv2.merge((l_enhanced, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return result


def apply_background_removal(img):
    original_h, original_w = img.shape[:2]
    max_dim = 400
    scale = min(max_dim / original_w, max_dim / original_h, 1.0)
    small = cv2.resize(img, (int(original_w * scale), int(original_h * scale)))

    mask = np.zeros(small.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    h, w = small.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))

    cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    mask2_full = cv2.resize(mask2, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    result = img * mask2_full[:, :, np.newaxis]
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    final = np.where(mask2_full[:, :, np.newaxis] == 1, result, white_bg)

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


def apply_blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)


def apply_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def apply_sharpen(img):
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


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
            elif operation == "contrast":
                result = apply_contrast(img)
            elif operation == "background":
                result = apply_background_removal(img)
            elif operation in ["erode", "dilate", "open", "close"]:
                result = apply_morphology(img, operation)
            elif operation in ["binary", "otsu", "adaptive"]:
                result = apply_threshold(img, operation)
            elif operation == "blur":
                result = apply_blur(img)
            elif operation == "edge":
                result = apply_edge_detection(img)
            elif operation == "sharpen":
                result = apply_sharpen(img)
            elif operation == "sepia":
                result = apply_sepia(img)
            elif operation == "colorize":
                result = apply_colorization(img)
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