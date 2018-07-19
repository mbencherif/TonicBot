import numpy as np

class DataReader():

    def __init__(self, glove_file_path):
        self.glove_file_path = glove_file_path

    def load_glove_model(self, gloveFile):
        print("Loading Glove Model...")
        f = open(self.glove_file_path, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print(f"Done. {len(model)} words loaded!")
        self.model = model