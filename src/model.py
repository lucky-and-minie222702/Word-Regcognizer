import numpy as np
import func
from func import  printProgress
from const import const
import random as rd
import time

class neuralNetwork:
    def save(self):
        np.savetxt(self.wihFile, self.wih)
        np.savetxt(self.whoFile, self.who)
        np.savetxt(self.wihBiasFile, self.wihBias)
        np.savetxt(self.whoBiasFile, self.whoBias)
        f = open(self.lrFile, 'w')
        print(self.lr, file=f)
        f.close()
        
    def loadData(self):
        self.wih = np.loadtxt(self.wihFile)
        self.who = np.loadtxt(self.whoFile)
        self.wihBias = np.loadtxt(self.wihBiasFile)
        self.whoBias = np.loadtxt(self.whoBiasFile)
        self.lr = float(open(self.lrFile, 'r').readline())
        
    def __init__(self, learningRateFile, wihFile, whoFile, wihBiasFile, whoBiasFile, init=True):
        self.inodes = const["inodes"]
        self.onodes = const["onodes"]
        self.hnodes = int(2/3 * self.inodes) + self.onodes
        self.deltaLr = const["deltaLr"]
        self.lrFile = learningRateFile
        self.sep = self.inodes
        # file paths
        self.wihFile = wihFile
        self.whoFile = whoFile
        self.wihBiasFile = wihBiasFile
        self.whoBiasFile = whoBiasFile
        # create link weights
        if init:
            print("Initializing link weights")
            self.wih = np.random.normal(0.0, 1.0, (self.hnodes, self.inodes))
            self.who = np.random.normal(0.0, 1.0, (self.onodes, self.hnodes))
            self.wihBias = np.zeros((self.hnodes, 1))
            self.whoBias = np.zeros((self.onodes, 1))
            self.lr = 0.0001
        else:
            print("Loading link weights...")
            self.loadData()
        
        self.acFunc = func.relu
        self.classifyFunc = func.softmax
        
    
    def singleTrain(self, inputLists, targetLists, autoSave = False):
        inp = np.array(inputLists, ndmin=2).T
        tar = np.array(targetLists, ndmin=2).T
        
        self.wihBias = self.wihBias.reshape((self.hnodes, 1))
        hinp = np.dot(self.wih, inp) + self.wihBias
        hout = self.acFunc(hinp)
        
        self.whoBias = self.whoBias.reshape((self.onodes, 1))
        finalInp = np.dot(self.who, hout) + self.whoBias
        finalOut = self.classifyFunc(finalInp)
        
        deltaY = (finalOut - tar)
        deltaHerr = np.dot(self.who.T, deltaY)
        deltaHerr[hout <= 0] = 0
        
        wihGradient = np.dot(deltaHerr, inp.T)
        wihBias = np.sum(deltaHerr, axis=0, keepdims=True)
        whoGradient = np.dot(deltaY, hout.T)
        whoBias = np.sum(deltaY, axis=0, keepdims=True)
        
        wihGradient += 0.01 * self.wih
        whoGradient += 0.01 * self.who
        
        self.who -= self.lr * whoGradient
        self.wih -= self.lr * wihGradient
        self.wihBias -= self.lr * wihBias
        self.whoBias -= self.lr * whoBias
        
        if None in self.whoBias:
            print("\n\nError")
            exit()
        
        if autoSave:
            self.save()
    
    def query(self, inputLists):
        inp = np.array(inputLists, ndmin=2).T
        self.wihBias = self.wihBias.reshape((self.hnodes, 1))
        hinp = np.dot(self.wih, inp) + self.wihBias
        hout = self.acFunc(hinp)
        self.whoBias = self.whoBias.reshape((self.onodes, 1))
        finalInp = np.dot(self.who, hout) + self.whoBias
        finalOut = self.classifyFunc(finalInp)
        return finalOut
    
    # User interface methods
    
    def train(self, limit, start, trainData, show=True):
        count = 0
        for record in trainData[start::]:
            if count == limit:
                break
            count += 1
            
            allVal = record.split(",")
            inp = (np.asarray(allVal[1::], dtype = np.float32) / 255 * 0.99) + 0.01
            tar = np.zeros(const["onodes"]) + 0.01
            tar[int(allVal[0])] = 0.99

            self.singleTrain(inp, tar)
            
            percent = int(count / limit * 100)
            if show:
                printProgress(percent, count, limit)
        if show:
            print()
        self.save()

    def test(self, limit, testData, savePath=const["savePath"] + const["saveFile"], show=True):
        correct = 0
        count = 0
        fullCorrect = [0 for _ in range(const["onodes"])]
        fullTotal = [0 for _ in range(const["onodes"])]
        
        f = open(savePath, "w")

        randTest = rd.choices(testData, k=limit)
        for record in randTest:
            count += 1
            allVal = record.split(',')
            correctLabel = int(allVal[0])
            inputs = (np.asarray(allVal[1:], dtype = np.float32) / 255.0 * 0.99) + 0.01
            outputs = self.query(inputs)
            label = np.argmax(outputs)
            fullTotal[correctLabel] += 1
            if (label == correctLabel):
                fullCorrect[correctLabel] += 1
                correct += 1
            print("CORRECT" if label == correctLabel else "WRONG  ", "Test cases:", count, "/", limit, "expected:", correctLabel, "output:", label,file=f)
            percent = int(count / limit * 100)
            if show:
                printProgress(percent, count, limit)
                
        print("-"*10+"SUMMARY"+"-"*10, file=f)
        for i in range(len(fullCorrect)):
            percent = 0
            if fullTotal[i] > 0:
                percent = (fullCorrect[i]/fullTotal[i])*100
                print(f"Label {i}: {fullCorrect[i]} / {fullTotal[i]} - {percent:.2f}%", file=f)
            else:
                print(f"Label {i}: doesn't appear", file=f)
                
        if show:
            print()
            print("-"*10+"SUMMARY"+"-"*10)
            for i in range(len(fullCorrect)):
                percent = 0
                if fullTotal[i] > 0:
                    percent = (fullCorrect[i]/fullTotal[i])*100
                    print(f"Label {i}: {fullCorrect[i]} / {fullTotal[i]} - {percent:.2f}%")
                else:
                    print(f"Label {i}: doesn't appear")
            print("DONE")
            print("Correct:", correct)
            print("Wrong:", limit - correct)
        f.close()
        print("Accuracy:", f"{np.trunc((correct / limit) * 10000) / 100}%")
        
    def trainmode(self, cur, epoch, trainData, testData, rest=0):
        # total 40 epochs for training
        for i in range(1, epoch+1):
            print("*** SECTION:", i)
            self.train(const["lrStep"], cur, trainData)
            self.test(100, testData, show=False)
            cur += const["lrStep"]
            print("Current:", cur)
            
            if i%const["lrIter"] == 0:
                self.lr *= const["deltaLr"]
                
            self.save()
            print("Resting...", end="\r")
            time.sleep(rest)