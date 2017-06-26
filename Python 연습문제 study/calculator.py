class Calculator:
    def __init__(self, value):
        self.value = value # value array로 들어오지!!
    def sum(self):
        sum = 0
        for x in self.value:
            sum += x
        self.sum = sum
        return sum
    def avg(self):
        avg = self.sum / len(self.value)
        return avg