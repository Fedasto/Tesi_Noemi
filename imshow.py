import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import re



# collect the frames with the glob.glob function
imgs=sorted(glob.glob('C:/Users/PC/Desktop/C0003/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

img_guess = cv.imread(imgs[8001])#[450:2150,800:2250]

plt.figure(figsize=(10,10))
plt.title('Write the initial target coordinates:')
plt.imshow(img_guess)
plt.show()