# Initialized

### Creating folders

```zsh
python3 src/perform.py init
```

### Restart and initialize neural network files

```zsh
python3 src/perform.py restart
```

### Download training data

Download 2 files from [here](https://drive.google.com/drive/folders/1aSQ8xArgX-79bnhSf2OmDm6HUcCNxyQE) to folder "traindata" and unzip them

### Install training data

```zsh
python3 src/perform.py install
```

# Command line arguments

### Loading training data

> "load" argument is necessary when train or test the model

```zsh
python3 src/perform.py load
```

### Infomation

```zsh
python3 src/perform.py info load
```

### Testing

Testing with 1000 test cases:

```zsh
python3 src/perform.py test 1000 load
```

### Training

Training for a total of 40 epochs, starting from index 0, after each epoch, the program rests 5 seconds before next epoch

> Every epoch the model faces 10000 images

```zsh
python3 src/perform.py test 40 0 5 load
```

(It is recommended to take some seconds after every epoch to rest yout cpu)
