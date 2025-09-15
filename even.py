num = [2,4,1,5,7,8,9]
even = []
odd = []
countEven = 0
countOdd = 0
for n in num:
    if n%2 == 0:
        countEven+=1
        even.append(n)
    else:
        countOdd += 1
        odd.append(n)
print("Even Numbers : ", countEven, ", Numbers are :",  even)
print("Odd Numbers : ", countOdd, ", Numbers are :", odd)

sum = 0
for n in num:
    sum += n
average = sum//len(num)


print("Average: ", average)