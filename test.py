from utils import parse_images
import glob, tf2onnx
from keras.models import load_model
import numpy as np
import onnx
import tensorflow as tf

test_files = glob.glob('data/test/*')

X_test = parse_images(test_files)

model = load_model('best.h5', compile=False)
model.output_names = ["output"]
y_pred = model.predict(X_test)
print(np.argmax(y_pred, axis=1))
# onnx_model, _ = tf2onnx.convert.from_keras(model)
# onnx.save(onnx_model, 'best.onnx')