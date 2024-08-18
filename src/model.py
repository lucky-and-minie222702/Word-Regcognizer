import numpy as np
import func
from func import  printProgress
from const import const
import random as rd
import time
from skimage.measure import block_reduce

class neuralNetwork:
    def save(self):
        np.savetxt(self.wihFile, self.wih)
        np.savetxt(self.whoFile, self.who)
        np.savetxt(self.wihBiasFile, self.wihBias)
        np.savetxt(self.whoBiasFile, self.whoBias)
        f = open(self.lrFile, 'w')
        print(self.lr, self.lrMax, self.lrMin, sep="\n", file=f)
        f.close()
        
    def loadData(self):
        self.wih = np.loadtxt(self.wihFile)
        self.who = np.loadtxt(self.whoFile)
        self.wihBias = np.loadtxt(self.wihBiasFile)
        self.whoBias = np.loadtxt(self.whoBiasFile)
        lr = open(self.lrFile, 'r').readlines()
        self.lr = float(lr[0])
        self.lrMax = float(lr[1])
        self.lrMin = float(lr[2])
        
    def __init__(self, learningRateFile, wihFile, whoFile, wihBiasFile, whoBiasFile, lrMax, lrMin, lrStep, init=True):
        self.inodes = const["inodes"]
        self.onodes = const["onodes"]
        self.hnodes = const["hnodes"]
        self.lrFile = learningRateFile
        self.sep = self.inodes
        self.lrMax = lrMax
        self.lrMin = lrMin
        self.lrStep = lrStep
        # file paths
        self.wihFile = wihFile
        self.whoFile = whoFile
        self.wihBiasFile = wihBiasFile
        self.whoBiasFile = whoBiasFile
        # create link weights
        if init:
            print("Initializing link weights")
            self.wih = np.random.normal(0.0, 1.0, (self.hnodes, self.inodes // 4))
            self.who = np.random.normal(0.0, 1.0, (self.onodes, self.hnodes))
            self.wihBias = np.zeros((self.hnodes, 1))
            self.whoBias = np.zeros((self.onodes, 1))
            self.lr = lrMax
        else:
            print("Loading link weights...")
            self.loadData()
        
        self.acFunc = func.relu
        self.classifyFunc = func.softmax
        
    
    def singleTrain(self, inputLists, targetLists, autoSave = False):
        inp = inputLists.reshape(28, 28)
        # pooling layers
        inp = block_reduce(inp, (2, 2), np.max)
        inp = inp.flatten()
        
        inp = np.array(inp, ndmin=2).T
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
        inp = inputLists.reshape(28, 28)
        # pooling layers
        inp = block_reduce(inp, (2, 2), np.max)
        inp = inp.flatten()
        
        inp = np.array(inp, ndmin=2).T
        self.wihBias = self.wihBias.reshape((self.hnodes, 1))
        # pooling layers
        hinp = np.dot(self.wih, inp) + self.wihBias
        hout = self.acFunc(hinp)
        self.whoBias = self.whoBias.reshape((self.onodes, 1))
        finalInp = np.dot(self.who, hout) + self.whoBias
        finalOut = self.classifyFunc(finalInp)
        return finalOut
    
    # User interface method
    
    def test(self, input, label, limit, savePath="test_log/log.txt", show=True):
        correct = 0
        count = 0
        fullCorrect = [0 for _ in range(const["onodes"])]
        fullTotal = [0 for _ in range(const["onodes"])]
        
        f = open(savePath, "w")

        randTest = rd.choices(range(len(input)), k=limit)
        for i in randTest:
            count += 1
            output= self.query(input[i])
            actualLabel = np.argmax(output)
            correctLabel = np.argmax(label[i])
            fullTotal[correctLabel] += 1
            if (actualLabel == correctLabel):
                fullCorrect[correctLabel] += 1
                correct += 1
            print("CORRECT" if actualLabel == correctLabel else "WRONG  ", "Test cases:", count, "/", limit, "expected:", correctLabel, "output:", actualLabel, file=f)
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
            print("-"*10+"FINAL"+"-"*10)
            print("Correct:", correct)
            print("Wrong:", limit - correct)
        f.close()
        print("Accuracy:", f"{np.trunc((correct / limit) * 10000) / 100}%")
        
    def train(self, input, label, epoch, cur, rest, frequentlyTest=True, testInput=[], testLabel=[]):
        count = 0
        for i in range(epoch):
            print("*** EPOCH:", i+1)
            count = 0
            for j in range(cur, cur+self.lrStep):
                count +=1
                
                cycle = np.floor(1 + count / self.lrStep)
                x = np.abs(count / self.lrStep - 2*cycle + 1)
                self.lr = self.lrMin + (self.lrMax - self.lrMin)*max(0.0, 1-x)
                
                self.singleTrain(input[j], label[j])
                percent = int(count / self.lrStep * 100)
                printProgress(percent, count, self.lrStep)
            print()
            self.lrMax *= 0.85
            self.lrMin *= 0.85
            self.save()
            cur += self.lrStep
            print("Current:", cur)
            
            if frequentlyTest:
                self.test(testInput, testLabel, 100, show=False)
            
            print("Resting...", end="\r")
            time.sleep(rest)
        self.save()