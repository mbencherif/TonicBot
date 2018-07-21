from data_reader import DataReader
import cProfile

def main():
  print("Hello World!")
  model = DataReader('3rd-party/GloVe/glove.840B.300d.txt')
  model.load_glove_model(10)

  # Print words to see how they are ordered
  for key, _ in model.items():
    print(f"{key}")

if __name__ == '__main__':
    cProfile.run('main()')