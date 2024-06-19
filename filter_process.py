import cv2
import numpy as np
import torch
def filter_large(image):
    if len(image.shape) == 4:
        image = image[0,0,:,:]
    numpy_array = image.numpy()
    binary_image = numpy_array.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个和输入图像大小相同的空白图像
    output_image = np.zeros_like(binary_image)
    if len(contours)<10:
        output_image = image
    else:
        for contour in contours:
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if len(approx) <= 5 and area>=14 and len(approx) >= 4:
                cv2.drawContours(output_image, [contour], -1, 255, -1)
        output_image = (torch.from_numpy(np.ascontiguousarray(output_image))>0).float()
    if len(output_image.shape) == 2:
        output_image = output_image.unsqueeze(0).unsqueeze(0)
    return output_image