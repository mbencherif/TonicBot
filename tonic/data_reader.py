import numpy as np

class DataReader():

    def __init__(self, glove_file_path):
        self.glove_file_path = glove_file_path

    def load_glove_model(self, line_count=1000):
        print("Loading Glove Model...")
        f = open(self.glove_file_path, 'r', encoding="utf8")
        model = {}
        for i, line in enumerate(f):
            if i >= line_count:
                break
                
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print(f"Done. {len(model)} words loaded!")
        self.model = model

    def items(self):
        return self.model.items()