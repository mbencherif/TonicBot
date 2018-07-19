import unittest

from model.data_reader import DataReader


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
        data_reader = DataReader("glove")
        self.assertNotEquals(data_reader, None)

if __name__ == '__main__':
    unittest.main()