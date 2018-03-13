class Game(object):
    def __init__(self, n):
        self.n = n

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1
