import numpy as np
import cv2
import argparse
from imutils import paths

labels = ["dog","cat","panda"]
np.random.seed(2)

W = np.random.randn(3,3072)
b = np.random.randn(3)

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to Image")
args = vars(ap.parse_args())

print("[INFO] Loading Image...")

orig = cv2.imread(args["image"])
image = cv2.resize(orig,(32,32),interpolation=cv2.INTER_AREA).flatten()

scores = W.dot(image) + b

for (label,score) in zip(labels,scores):
    print("[INFO] {}:{:.2f}".format(label,score))


cv2.putText(orig,"Label:{}".format(labels[np.argmax(scores)]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

cv2.imshow("Image",orig)
cv2.waitKey(0)
