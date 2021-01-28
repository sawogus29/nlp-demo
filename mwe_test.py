import unittest
from components.mwe import get_model_and_tokenizer, validate_iob, align_labels

class Test_MWE(unittest.TestCase):
    def test_validate_iob(self):
        x = ['B','O','I','O','B','o','I','I','O']
        correct_y = ['O','O','O','O','B','o','I','I','O']
        y = validate_iob(x)

        self.assertEqual(y, correct_y)
    
    def test_align_labels(self):
        labels = ['B', 'I', 'I', 'O', 'O', 'B', 'o', 'I', 'I', 'O', 'O']
        alignments = [['B', 'I'],['I'], ['O'], ['O'], ['B'], ['o', 'I'], ['I', 'O'], ['O']]
        correct_aligned_labels = ['B', 'I', 'O', 'O', 'B', 'I', 'I', 'O']
        aligned_labels = align_labels(labels, alignments)
        
        self.assertEqual(aligned_labels, correct_aligned_labels)




    # def test_get_model_and_tokenizer(self):
    #     model, tknz = get_model_and_tokenizer()
    #     self.assert_(True)

if __name__ == '__main__':
    unittest.main()