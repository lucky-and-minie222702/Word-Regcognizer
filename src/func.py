import numpy as np
from scipy.special import softmax as sft
import os
from os.path import dirname, basename
import shutil        
from const import const
import sys

def relu(inputs):
    return np.maximum(inputs, 0)

def softmax(inputs):
    return sft(inputs)

def emptyDir(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            elif os.path.isfile(file_path):
                os.remove(file_path)
            print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")

def printProgress(percent, count, limit):
    loaded = "=" * (percent//2)
    unloaded = " " * (50 - (percent//2))
    print(f"{percent:3d}% [{loaded}{unloaded}]", "Inputs:", count, "/", limit, end="\r")
    sys.stdout.flush()