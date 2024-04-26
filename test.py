import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession('fake_detect.onnx')
img = cv2.imread("data/test/image.jpeg")
img = cv2.resize(img, (128, 128))
img_tensor = np.expand_dims(np.array(img) / 255.0, 0)
print(img_tensor.shape)
img_array = np.ascontiguousarray(np.expand_dims(np.array(img) / 255.0, 0)).astype(np.float32)
onnx_inputs = {session.get_inputs()[0].name: img_array}
onnx_outputs = session.run(None, onnx_inputs)[0]
print(np.argmax(onnx_outputs[0]))