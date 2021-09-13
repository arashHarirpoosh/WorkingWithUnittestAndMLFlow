import unittest
from CIFAR10 import loadDataset


class TestShape(unittest.TestCase):
    # @classmethod
    # def setUp(cls) -> None:
    #     x_train, x_val, x_test, y_train, y_val, y_test = loadDataset()
    def test_dim(self):
        x_train, x_val, x_test, y_train, y_val, y_test = loadDataset()
        check_x_train = [1 if x.shape[2] == 3 else 0 for x in x_train]
        check_x_test = [1 if x.shape[2] == 3 else 0 for x in x_test]
        check_x_val = [1 if x.shape[2] == 3 else 0 for x in x_val]

        check_y_train = [1 if y.shape[0] == 10 else 0 for y in y_train]
        check_y_test = [1 if y.shape[0] == 10 else 0 for y in y_test]
        check_y_val = [1 if y.shape[0] == 10 else 0 for y in y_val]

        self.assertEqual(sum(check_x_train), x_train.shape[0], "Third dimension should be 3 for all train data")
        self.assertEqual(sum(check_x_test), x_test.shape[0], "Third dimension should be 3 for all test data")
        self.assertEqual(sum(check_x_val), x_val.shape[0], "Third dimension should be 3 for all validation data")

        self.assertEqual(sum(check_y_train), y_train.shape[0], "All labels of the train data should be one hot encoded")
        self.assertEqual(sum(check_y_test), y_test.shape[0], "All labels of the test data should be one hot encoded")
        self.assertEqual(sum(check_y_val), y_val.shape[0], "All labels of the validation data should be one hot encoded")


if __name__ == '__main__':
    unittest.main()
