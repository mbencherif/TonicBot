import unittest

from tonic.data_reader import DataReader

class TestDataReader(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_load_glove_model(self):
        word_count = 50000
        data_reader = DataReader("3rd-party\\GloVe\\glove.840B.300d.txt")
        data_reader.load_glove_model(word_count)

        self.assertEqual(word_count, len(data_reader.items()))

if __name__ == '__main__':
    unittest.main()