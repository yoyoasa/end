
# write a program to count characters in a string
st = "AmmarAdil"
count = {}
for a in st:
    if a in count:
        count[a]+=1
    else:
        count[a] = 1
print('Count', count)


# write a program to print count of vowels in a string
st = "ammaradil"
vowle = ['a', 'e', 'i', 'o', 'u']
count = 0

for s in st:
    if s in vowle:
        count = count+1

print("Count", count)


# write program to convert string to upper case
st = "ammar adil"

upper_st = st.upper()
print("Upper Case", upper_st)


# write program to convert string to lower case
st = "AMMAR ADIL"

lower_st = st.lower()
print("Lower Case", lower_st)


# write a program to find union of 2 arrays
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

union_both = a.union(b)
print("Union", union_both)


# write a program to find intersection
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

intersection_both = a.intersection(b)
print("Intersection", intersection_both)


# write a program to create print array in beautiful format
a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

for i in a:
    row = '|'
    for b in i:
        row = row + ' ' + str(b)
    print(row + ' ' + '|')


# write a program to create zero matrix
rows = 2
cols = 3
M = []
while len(M) < rows:
    M.append([])
    while len(M[-1]) < cols:
        M[-1].append(0.0)

print("Zero Matrix")
for i in range(rows):
    row = '|'
    for b in range(cols):
        row = row + ' ' + str(M[i][b])
    print(row + ' ' + '|')


# write a program to create identity matrix with dimension provided
dim = 3
M = []
while len(M) < dim:
    M.append([])
    while len(M[-1]) < dim:
        M[-1].append(0.0)

for i in range(dim):
    M[i][i] = 1.0

print('Identity Matrix')
for i in range(dim):
    row = '|'
    for b in range(dim):
        row = row + ' ' + str(M[i][b])
    print(row + ' ' + '|')


# Write a program to copy a given array
M = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
rows = len(M)
cols = len(M[0])

MC = []
while len(MC) < rows:
    MC.append([])
    while len(MC[-1]) < cols:
        MC[-1].append(0.0)

for i in range(rows):
    for j in range(cols):
        MC[i][j] = M[i][j]

print("Copied Array")
for i in range(rows):
    row = '|'
    for b in range(cols):
        row = row + ' ' + str(MC[i][b])
    print(row + ' ' + '|')


# write a program to transpose a matrix
M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

if not isinstance(M[0], list):
    M = [M]

rows = len(M)
cols = len(M[0])

MT = []
while len(MT) < dim:
    MT.append([])
    while len(MT[-1]) < dim:
        MT[-1].append(0.0)

for i in range(rows):
    for j in range(cols):
        MT[j][i] = M[i][j]

print("Transpose Array")
for i in range(rows):
    row = '|'
    for b in range(cols):
        row = row + ' ' + str(MT[i][b])
    print(row + ' ' + '|')


# write a program to add two matrix
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

rowsA = len(A)
colsA = len(A[0])
rowsB = len(B)
colsB = len(B[0])
if rowsA != rowsB or colsA != colsB:
    raise ArithmeticError('Matrices are NOT the same size.')

C = []
while len(C) < rowsA:
    C.append([])
    while len(C[-1]) < colsB:
        C[-1].append(0.0)

for i in range(rowsA):
    for j in range(colsB):
        C[i][j] = A[i][j] + B[i][j]

print("Added Array")
for i in range(rowsA):
    row = '|'
    for b in range(colsA):
        row = row + ' ' + str(C[i][b])
    print(row + ' ' + '|')


# write a program to subtract two matrix
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

rowsA = len(A)
colsA = len(A[0])
rowsB = len(B)
colsB = len(B[0])
if rowsA != rowsB or colsA != colsB:
    raise ArithmeticError('Matrices are NOT the same size.')

C = []
while len(C) < rowsA:
    C.append([])
    while len(C[-1]) < colsB:
        C[-1].append(0.0)

for i in range(rowsA):
    for j in range(colsB):
        C[i][j] = A[i][j] - B[i][j]

print("Subtracted Array")
for i in range(rowsA):
    row = '|'
    for b in range(colsA):
        row = row + ' ' + str(C[i][b])
    print(row + ' ' + '|')


# write a program to multiply two matrix

rowsA = len(A)
colsA = len(A[0])
rowsB = len(B)
colsB = len(B[0])

if colsA != rowsB:
    raise ArithmeticError('Number of A columns must equal number of B rows.')

C = []
while len(C) < rowsA:
    C.append([])
    while len(C[-1]) < colsB:
        C[-1].append(0.0)

for i in range(rowsA):
    for j in range(colsB):
        total = 0
        for ii in range(colsA):
            total += A[i][ii] * B[ii][j]
        C[i][j] = total

print("Multiplied Array")
for i in range(rowsA):
    row = '|'
    for b in range(colsA):
        row = row + ' ' + str(C[i][b])
    print(row + ' ' + '|')


# write a program to join all items in a tuple into a string, using a hash character as separator
myTuple = ("John", "Peter", "Vicky")
x = "#".join(myTuple)
print(x)


# write a program to remove spaces at the beginning and at the end of the string
txt = "     banana     "
x = txt.strip()
print("of all fruits", x, "is my favorite")


# write a program to remove the leading and trailing characters
txt = ",,,,,rrttgg.....banana....rrr"
x = txt.strip(",.grt")
print(x)


# write a program to split a string into a list where each line is a list item
txt = "Thank you for the music\nWelcome to the jungle"
x = txt.splitlines()
print(x)


# write a program to find index of a word in given string
txt = "Hello, welcome to my world."
x = txt.index("welcome")
print(x)


# write a program to find ceil of a number
import math

number = 34.564
ce = math.ceil(number)
print('Ceil', ce)


# write a program to find absoluute number of a given number
import math

number = 34.564
fa = math.fabs(number)
print('Fabs', fa)


# write a program to find factorinal of a number
import math

number = 8
fa = math.factorial(number)
print('Factorial', fa)

# write a program to find exponential of a number
import math

number = 3

print('Exponential', math.exp(number))


# write a program to find log of a number
import math

num = 5
base = 7

print("Log_x_b", math.log(num, base))


# write a program to find cosine of a number
import math

num = 45
print("Cosine", math.cos(num))


# write a program to find sin of a number
import math

num = 45
print("Sin", math.sin(num))


# write a program to find tangent of a number
import math

num = 45
print("Tangent", math.tan(num))


# Write a program to print bit wise AND of two numbers
a = 60            # 60 = 0011 1100
b = 13            # 13 = 0000 1101

c = a & b        # 12 = 0000 1100
print("AND", c)


# Write a program to print bit wise OR of two numbers
a = 60
b = 13

c = a | b
print("OR", c)


# Write a program to print bit wise XOR of two numbers
a = 60
b = 13

c = a ^ b
print("XOR", c)


# Write a program to calculate Binary Ones Complement of a number
a = 60

c = ~a
print("Binary Ones Complement", c)


# write a program to Binary Left Shift a number
c = a << 2
print("Binary Left Shift", c)


# write a program to Binary Right Shift a number
c = a >> 2
print("Binary Right Shift", c)

