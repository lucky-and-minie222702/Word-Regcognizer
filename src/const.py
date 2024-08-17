const = {
    "lrStep": 10000, 
    "lrIter": 1,
    "inodes": 784,
    "onodes": 26 + 10,
    "lrMax": 0.001,
    "lrMin": 0.000001,
    "learningRateFile": "learning_rate.txt",
    "trainPath": "traindata",
    "trainFile": "traindata/trainData.csv",
    "testFile": "traindata/testData.csv",
    "dataPath": "data",
    "wihFile": "neuron_wih.txt",
    "whoFile": "neuron_who.txt",
    "biasWihFile": "bias_wih.txt",
    "biasWhoFile": "bias_who.txt",
    "savePath": "test_log/",
    "saveFile": "log.txt",
}

def init(path = const["dataPath"]):
    const["dataPath"] = path
    const["learningRateFile"] = const["dataPath"] + "/" + const["learningRateFile"]
    const["wihFile"] = const["dataPath"] + "/" + const["wihFile"]
    const["whoFile"] = const["dataPath"] + "/" + const["whoFile"]
    const["biasWihFile"] = const["dataPath"] + "/" + const["biasWihFile"]
    const["biasWhoFile"] = const["dataPath"] + "/" + const["biasWhoFile"]