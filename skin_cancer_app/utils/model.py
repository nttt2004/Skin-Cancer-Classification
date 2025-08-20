import numpy as np

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def load_model():
    # Demo: return a dummy handle
    return {"name": "demo_random_model", "classes": CLASSES}

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict_image(model, img_path):
    # Demo prediction: deterministic pseudo-random based on filename for stability
    seed = sum(bytearray(img_path.encode("utf-8"))) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=len(CLASSES))
    probs = softmax(logits)
    return {cls: float(p) for cls, p in zip(CLASSES, probs)}
