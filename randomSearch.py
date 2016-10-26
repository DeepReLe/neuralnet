import random


def uniform(a, b):
    return random.random()*b+a

def normal(mean, std):
    return mean

def seed(seed_in):
    return

class Experiment:

    def __init__(self, name, dictIn):
        self.name = name
        self.dict = dictIn

    def seed():
        return 0



    def printDict():
        for i in self.dict:
            print i
