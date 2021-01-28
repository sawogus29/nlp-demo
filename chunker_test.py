import unittest
import pipelines

class Test_chunker(unittest.TestCase):
    def test_generate_html(self):
        nlp = pipelines.get_nlp()
        doc = nlp("It is a fact that he loves her.")
        y = doc._.generate_html(show_word=False)
        correct_y = ""

        self.assertEqual(y, correct_y)

if __name__ == '__main__':
    unittest.main()