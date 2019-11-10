import cv2
import mahotas
from mahotas.features import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

bins = 8
seed = 77
DATA_DIR = "./data"
size=(60, 100)

def mask(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    def edgedetect (channel):
        stype = cv2.CV_16S
        sobelX = cv2.Sobel(channel, stype, 1, 0)
        sobelY = cv2.Sobel(channel, stype, 0, 1)
        sobel = np.hypot(sobelX, sobelY)
        sobel[sobel > 255] = 255
        return sobel
    edgeImg = np.max( np.array([ edgedetect(blurred[:,:, 0]), edgedetect(blurred[:,:, 1]), edgedetect(blurred[:,:, 2]) ]), axis=0 )
    mean = np.mean(edgeImg);
    edgeImg[edgeImg <= mean] = 0;
    def findSignificantContours (img, edgeImg):
        contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        level1 = []
        for i, tupl in enumerate(heirarchy[0]):
            if tupl[3] == -1:
                tupl = np.insert(tupl, 0, [i])
                level1.append(tupl)
        significant = []
        tooSmall = edgeImg.size * 10 / 100
        for tupl in level1:
            contour = contours[tupl[0]];
            area = cv2.contourArea(contour)
            if area > tooSmall:
                significant.append([contour, area])
                #cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
        significant.sort(key=lambda x: x[1])
        return [x[0] for x in significant]
    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    significant = findSignificantContours(img, edgeImg_8u)
    mask = edgeImg.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    mask = np.logical_not(mask)
    img[mask] = 0
    return img

# Hu Moments
def Hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture
def Haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def LBP(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    spoints = lbp(gray, 64, 4)
    return spoints

# Color Histogram
def Histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten() 

def preprocess(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.resize(image, size)
    #image = quantize(image)
    #image = mask(image)
    return image

def get_feature(image):
    image = preprocess(image)
    hu_moments = Hu_moments(image)
    haralick = Haralick(image)
    histogram = Histogram(image)
    #sr = LBP(image)
    return np.hstack([histogram, haralick, hu_moments])

def quantize(img, k = 8):
    (h, w) = img.shape[:2]
    image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    color = clt.cluster_centers_.astype("uint8")[0]
    #return clt.cluster_centers_.flatten()
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    #quant[quant == clt.cluster_centers_[1]] = [0,0,0]
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant
    # display the images and wait for a keypress
    #return np.hstack([image, quant])

def color(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors, count

if __name__ == "__main__":
    img = cv2.imread("/home/mint/Workspace/UMA/ML/Core/images/11197.png")
    img = preprocess(img)
    print (color(img))
    cv2.imshow("IMG", img)
    cv2.waitKey()
