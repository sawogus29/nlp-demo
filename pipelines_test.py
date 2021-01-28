import unittest
import pipelines

class Test_pipelines(unittest.TestCase):
    def test_pipe(self):
        nlp = pipelines.get_nlp()
        x = "He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a fish. In the first forty days a boy had been with him. But after forty days without a fish the boy's parents had told him that the old man was now definitely and finally salao, which is the worst form of unlucky, and the boy had gone at their orders in another boat which caught three good fish the first week. It made the boy sad to see the old man come in each day with his skiff empty and he always went down to help him carry either the coiled lines or the gaff and harpoon and the sail that was furled around the mast. The sail was patched with flour sacks and, furled, it looked like the flag of permanent defeat.  "
        y = nlp(x)
        correct_y = ['O','O','O','O','B','o','I','I','O']
        y = validate_iob(x)

        self.assertEqual(y, correct_y)

if __name__ == '__main__':
    unittest.main()