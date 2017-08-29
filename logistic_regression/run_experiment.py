import os

for i in range(1000,1100):
    #print("For seed {}".format(i))
    os.system("./logistic_regression.py ../../csci4360-fa17/assignments/assignment1/writeup/data/train.data ../../csci4360-fa17/assignments/assignment1/writeup/data/train.label ../../csci4360-fa17/assignments/assignment1/writeup/data/test_partial.data ../../csci4360-fa17/assignments/assignment1/writeup/data/test_partial.label "+str(i))
    #print()

