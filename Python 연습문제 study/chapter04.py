# Q1.
def fib(n):
    if n == 0 : return 0
    if n == 1 : return 1
    return fib(n-1) + fib(n-2)
a0 = fib(0)
a1 = fib(1)
a2 = fib(2)
a3 = fib(3)
a4 = fib(4)
a5 = fib(5)
a6 = fib(6)
a7 = fib(7)
a8 = fib(8)
a9 = fib(9)
a10 = fib(10)
print(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10)

# Q2.
f = open("sample.txt", "r")
lines = f.readlines()
sum = 0
for line in lines:
    data = line
    sum = sum + int(data)
avg = sum/len(lines)
f.close()
# print(sum)
# print(avg)
with open("result.txt", "w") as t:
    data = str(sum) + "\n" + str(avg)
    t.write(data)
