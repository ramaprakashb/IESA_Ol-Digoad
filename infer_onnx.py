import argparse
import numpy as np
import cv2
import onnxruntime as ort

def load_labels(path="labels.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def preprocess(image_path, size=224):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = img[None, None, :, :]  # (1,1,H,W)
    return img

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Model/Ol-Digoad_IESA.onnx", help="Path to ONNX model")
    ap.add_argument("--image", required=True, help="Path to SEM image")
    ap.add_argument("--labels", default="labels.txt", help="Path to labels.txt")
    args = ap.parse_args()

    labels = load_labels(args.labels)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    x = preprocess(args.image, size=224)
    out = sess.run(None, {input_name: x})[0][0]  # shape: (num_classes,)

    probs = softmax(out)
    idx = int(np.argmax(probs))
    print("Predicted:", labels[idx] if idx < len(labels) else idx)
    print("Confidence:", float(probs[idx]))

if __name__ == "__main__":
    main()
