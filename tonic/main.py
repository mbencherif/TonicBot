from data_reader import DataReader
import cProfile
import os

def main():
  print("Hello World!")
  model = DataReader('3rd-party/GloVe/glove.840B.300d.txt')
  model.load_glove_model(10)

  # Print words to see how they are ordered
  for key, value in model.items():
    print(f"{key}: {value}")

def process_labels():
  imdb_dir = 'data/aclImdb'
  train_dir = os.path.join(imdb_dir, 'train')
  labels = []
  texts = []
  for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
  for fname in os.listdir(dir_name):
    if fname[-4:] == '.txt':
      f = open(os.path.join(dir_name, fname))
      texts.append(f.read())
      f.close()
    if label_type == 'neg':
      labels.append(0)
    else:
      labels.append(1)

if __name__ == '__main__':
    cProfile.run('main()')