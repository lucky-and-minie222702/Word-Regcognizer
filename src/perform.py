import numpy as np
import random as rd
from const import const
from model import neuralNetwork
import sys
import os
from func import  printProgress
import random as rd
import cv2
import time
from imutils.contours import sort_contours
import imutils

trainInput = []
trainLabel = []
testInput = []
testLabel = []

if "init" in sys.argv:
    print("Initializing...")
    os.mkdir("test_log/")
    os.mkdir("traindata/")
    print("DONE")
    exit()

if "install" in sys.argv:
    prompt = input("Type \"yes\" to continue: ")
    if prompt != "yes":
        print("Instalation cancelled")
        exit()
    print("Processing train data...")
    rawData = open("traindata/trainData.csv", "r").readlines()
    count = 0
    limit = len(rawData)
    for record in rawData:
        count += 1
        allVal = record.split(",")
        inp = (np.asarray(allVal[1::], dtype = np.float32) / 255 * 0.99) + 0.01
        tar = np.zeros(const["onodes"]) + 0.01
        tar[int(allVal[0])] = 0.99
        
        trainInput.append(inp)
        trainLabel.append(tar)
        
        percent = int(count / limit * 100)
        printProgress(percent, count, limit)

    print()
    print("Processing test data...")
    rawData = open("traindata/testData.csv", "r").readlines()
    count = 0
    limit = len(rawData)
    for record in rawData:
        count += 1
        allVal = record.split(",")
        inp = (np.asarray(allVal[1::], dtype = np.float32) / 255 * 0.99) + 0.01
        tar = np.zeros(const["onodes"]) + 0.01
        tar[int(allVal[0])] = 0.99
        
        testInput.append(inp)
        testLabel.append(tar)
        
        percent = int(count / limit * 100)
        printProgress(percent, count, limit)
    
    rawData = []
    print("\nSaving data...")
    np.save("traindata/trainInput", trainInput)
    np.save("traindata/trainLabel", trainLabel)
    np.save("traindata/testInput", testInput)
    np.save("traindata/testLabel", testLabel)
    print("Installation done!")
    exit()
    
if "load" in sys.argv:
    trainInput = np.load("traindata/trainInput.npy")
    trainLabel = np.load("traindata/trainLabel.npy")
    testInput = np.load("traindata/testInput.npy")
    testLabel = np.load("traindata/testLabel.npy")
    
model = neuralNetwork(
    const["learningRateFile"],
    const["wihFile"],
    const["whoFile"],
    const["biasWihFile"],
    const["biasWhoFile"],
    const["lrMax"],
    const["lrMin"],
    const["lrStep"],
)

if "restart" in sys.argv:
    prompt = input("Restart destroys every current training data in default path!\nAre you sure you want to restart (yes=yes, no=everything else)?: ")
    if prompt == "yes":
        model.save()
    else:
        print("Restart cancelled!")
    exit()

if not ("load" in sys.argv):
    print("*** WARNING: You have not loaded the model yet! (add \"load\" to your command to load the model)")
    time.sleep(3)

print("Loading link weights...")
model.loadData()

trainSize = len(trainInput)
testSize = len(testInput)

if "info" in sys.argv:
    print("-"*10+"INFOMARTION"+"-"*10)
    print("Train size:", trainSize, " - Test size: ", testSize)
    exit()

# 0 1 2 ... 7 8 9 a b c ... x y z 

def predict():
    img = cv2.imread(sys.argv[2])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    chars = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        dX = int(max(0, 28 - tW) / 2.0)
        dY = int(max(0, 28 - tH) / 2.0)
        
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0))
        padded = cv2.resize(padded, (28, 28))
        padded = (padded.astype("float32") / 255.0) * 0.99 + 0.01
        chars.append(padded)

    img = np.ndarray.flatten(np.array(chars[0]))

    ans = model.query(img)
    if "full" in sys.argv:
        for i, val in enumerate(ans):
            if i > 9:
                i = chr(ord('a') + i - 10) + " - " + str(i)
            print("Label:", i, f"- {val[0]*100:.2f}%")
    ans = np.argmax(ans)
    if ans > 9:
        ans = chr(ord('a') + ans - 10)
    print("Final answer:", ans)

if "train" in sys.argv:
    prompt = input("Type \"yes\" to continue: ")
    if prompt != "yes":
        print("Training cancelled")
        exit()
    epoch = int(sys.argv[2])
    cur = int(sys.argv[3])
    rest = float(sys.argv[4])
    model.train(trainInput, trainLabel, epoch, cur, rest, True, testInput, testLabel)
elif "test" in sys.argv:
    limit = int(sys.argv[2])
    model.test(testInput, testLabel, limit)
    print("Full result saved to \"test_log/log.txt\"")
elif "predict" in sys.argv:
    predict() 
