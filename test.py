from maskrcnn_module import inference
import numpy as np

maskrcnn = inference.Maskrcnn_Module('m1.pth', 'm2.pth')

rgb = np.load("rgb.npy")
depth = np.load("depth.npy")
print(depth.shape)
print(rgb.shape)
cam_intr = [432.97146127, 432.97146127, 240, 320]

maskrcnn_output = maskrcnn.run_inference_preview(rgb,depth,cam_intr)

print(maskrcnn_output)