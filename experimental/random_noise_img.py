import cv2
import numpy as np
import time

while True:
    img = np.uint8(np.random.rand(500,500,3)*255)
    img1 = np.uint8(np.random.rand(50,50,3)*255)
    img1 = cv2.resize(img1,(500,500))
    cv2.imshow("random",img)
    cv2.imshow("random1",img1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    time.sleep(2)
