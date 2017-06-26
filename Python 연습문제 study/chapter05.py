# Q1.
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

cal1 = Calculator([1,2,3,4,5])
print(cal1.sum())
print(cal1.avg())


# Q2.

from calculator import Calculator
cal1 = Calculator([1,2,3,4,5])
print(cal1.sum())