Device: cuda
Data Set Size 7015
Valid Data Size 701
Unique tokens in Question vocabulary: 1451
Unique tokens in Answer vocabulary: 3299
BATCH_SIZE: 32
INPUT_DIM: 1451
OUTPUT_DIM: 3299
HID_DIM: 300
ENC_LAYERS: 4
DEC_LAYERS: 4
ENC_HEADS: 5
DEC_HEADS: 5
ENC_PF_DIM: 512
DEC_PF_DIM: 512
ENC_DROPOUT: 0.1
DEC_DROPOUT: 0.1
SRC_PAD_IDX: 1
TRG_PAD_IDX: 1
The model has 9,318,495 trainable parameters
Pretrained embedding dimension: torch.Size([3299, 300])
LEARNING_RATE: 0.0005
MAX_LR: 0.001
N_EPOCHS: 24
CLIP: 1
STEPS_PER_EPOCH: 220

Learning Rate: 4e-05
Epoch: 01 | Time: 0m 15s
	Train Loss: 4.714 | Train PPL: 111.482
	 Val. Loss: 2.634 |  Val. PPL:  13.923

Learning Rate: 0.0001734175615919141
Epoch: 02 | Time: 0m 14s
	Train Loss: 2.548 | Train PPL:  12.787
	 Val. Loss: 2.080 |  Val. PPL:   8.004

Learning Rate: 0.0003068351231838282
Epoch: 03 | Time: 0m 14s
	Train Loss: 2.130 | Train PPL:   8.417
	 Val. Loss: 1.785 |  Val. PPL:   5.960

Learning Rate: 0.00044025268477574227
Epoch: 04 | Time: 0m 14s
	Train Loss: 1.886 | Train PPL:   6.595
	 Val. Loss: 1.597 |  Val. PPL:   4.938

Learning Rate: 0.0005736702463676564
Epoch: 05 | Time: 0m 14s
	Train Loss: 1.711 | Train PPL:   5.532
	 Val. Loss: 1.476 |  Val. PPL:   4.374

Learning Rate: 0.0007070878079595705
Epoch: 06 | Time: 0m 13s
	Train Loss: 1.593 | Train PPL:   4.919
	 Val. Loss: 1.313 |  Val. PPL:   3.716

Learning Rate: 0.0008405053695514845
Epoch: 07 | Time: 0m 13s
	Train Loss: 1.505 | Train PPL:   4.502
	 Val. Loss: 1.250 |  Val. PPL:   3.489

Learning Rate: 0.0009739229311433986
Epoch: 08 | Time: 0m 13s
	Train Loss: 1.418 | Train PPL:   4.130
	 Val. Loss: 1.131 |  Val. PPL:   3.097

Learning Rate: 0.0009521105811688312
Epoch: 09 | Time: 0m 13s
	Train Loss: 1.294 | Train PPL:   3.646
	 Val. Loss: 1.007 |  Val. PPL:   2.738

Learning Rate: 0.0008925870097402597
Epoch: 10 | Time: 0m 13s
	Train Loss: 1.174 | Train PPL:   3.234
	 Val. Loss: 0.914 |  Val. PPL:   2.494

Learning Rate: 0.0008330634383116884
Epoch: 11 | Time: 0m 13s
	Train Loss: 1.077 | Train PPL:   2.934
	 Val. Loss: 0.826 |  Val. PPL:   2.284

Learning Rate: 0.0007735398668831169
Epoch: 12 | Time: 0m 13s
	Train Loss: 0.990 | Train PPL:   2.691
	 Val. Loss: 0.764 |  Val. PPL:   2.146

Learning Rate: 0.0007140162954545454
Epoch: 13 | Time: 0m 13s
	Train Loss: 0.925 | Train PPL:   2.521
	 Val. Loss: 0.697 |  Val. PPL:   2.008

Learning Rate: 0.0006544927240259741
Epoch: 14 | Time: 0m 14s
	Train Loss: 0.867 | Train PPL:   2.379
	 Val. Loss: 0.655 |  Val. PPL:   1.924

Learning Rate: 0.0005949691525974027
Epoch: 15 | Time: 0m 14s
	Train Loss: 0.809 | Train PPL:   2.245
	 Val. Loss: 0.584 |  Val. PPL:   1.794

Learning Rate: 0.0005354455811688312
Epoch: 16 | Time: 0m 15s
	Train Loss: 0.752 | Train PPL:   2.121
	 Val. Loss: 0.532 |  Val. PPL:   1.702

Learning Rate: 0.00047592200974025986
Epoch: 17 | Time: 0m 13s
	Train Loss: 0.698 | Train PPL:   2.011
	 Val. Loss: 0.474 |  Val. PPL:   1.607

Learning Rate: 0.0004163984383116884
Epoch: 18 | Time: 0m 13s
	Train Loss: 0.654 | Train PPL:   1.923
	 Val. Loss: 0.445 |  Val. PPL:   1.561

Learning Rate: 0.0003568748668831169
Epoch: 19 | Time: 0m 13s
	Train Loss: 0.608 | Train PPL:   1.836
	 Val. Loss: 0.399 |  Val. PPL:   1.490

Learning Rate: 0.00029735129545454555
Epoch: 20 | Time: 0m 13s
	Train Loss: 0.571 | Train PPL:   1.770
	 Val. Loss: 0.363 |  Val. PPL:   1.438

Learning Rate: 0.00023782772402597407
Epoch: 21 | Time: 0m 13s
	Train Loss: 0.529 | Train PPL:   1.697
	 Val. Loss: 0.338 |  Val. PPL:   1.402

Learning Rate: 0.0001783041525974026
Epoch: 22 | Time: 0m 15s
	Train Loss: 0.500 | Train PPL:   1.649
	 Val. Loss: 0.308 |  Val. PPL:   1.361

Learning Rate: 0.00011878058116883124
Epoch: 23 | Time: 0m 15s
	Train Loss: 0.472 | Train PPL:   1.603
	 Val. Loss: 0.294 |  Val. PPL:   1.341

Learning Rate: 5.9257009740259765e-05
Epoch: 24 | Time: 0m 14s
	Train Loss: 0.452 | Train PPL:   1.571
	 Val. Loss: 0.286 |  Val. PPL:   1.331
Best Model: capstone-model-23-0.2859172204678709.pt
Enter your question or type 'exit': addition of two number
#### addition of two number ####
------------------------------------------------------------
def add(num1,num2):
	sum=num1+num2
	return sum
------------------------------------------------------------
Enter your question or type 'exit': multiplication of two number
#### multiplication of two number ####
------------------------------------------------------------
def is_power_of_two(num1,num2):
	sum=num1+num2
	return sum
------------------------------------------------------------
Enter your question or type 'exit': multiply
#### multiply ####
------------------------------------------------------------
def multiply(x,y):
	if x<y:
		return -multiply(x,-y)
	else:
		return x*multiply(x,y-y)
------------------------------------------------------------
Enter your question or type 'exit': substract
#### substract ####
------------------------------------------------------------
def <unk>(n):
	return n*(n-1)
------------------------------------------------------------
Enter your question or type 'exit': substring 
#### substring ####
------------------------------------------------------------
str1="abc4234afde"
digitcount=0
for i in range(0,len(str1)):
	char=str1[i]
	if(char.isdigit()):
		digitcount+=1
print('number total lower case:',digitcount)
------------------------------------------------------------
Enter your question or type 'exit': write a python program to add two numbers
#### write a python program to add two numbers ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'sum:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': write Program to Add Two Matrices
#### write Program to Add Two Matrices ####
------------------------------------------------------------
a=[2,3,8,9,2,4,6]
b=[]
a,b=b[a+b for a in b for b in range(a,b)]
print(a)
------------------------------------------------------------
Enter your question or type 'exit': write Program to Transpose a Matrix
#### write Program to Transpose a Matrix ####
------------------------------------------------------------
x=[12,7],
	[4,5],
	[3,8],
	[4,5,5]]]
result=[[[[0,0,0],0],
		 [0,0,0]]
for i in range(len(x)):
	for j in range(len(x[0])):
		result[j][j][j][j][j][j][j][j][j][j]
for r in result:
	print(r)
------------------------------------------------------------
Enter your question or type 'exit': write Program to Multiply Two Matrice
#### write Program to Multiply Two Matrice ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'sum:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': write Program to Sort Words in Alphabetic Order
#### write Program to Sort Words in Alphabetic Order ####
------------------------------------------------------------
my_str="hello this is an example with cased letters"
words=[word.lower()for word in my_str.split()]
words.sort()
print("the sorted words are:")
for word in words:
	print(word)
------------------------------------------------------------
Enter your question or type 'exit': write Program to Illustrate Different Set Operations
#### write Program to Illustrate Different Set Operations ####
------------------------------------------------------------
def symmetric_diff_sets(a,b):
	return a & b)
------------------------------------------------------------
Enter your question or type 'exit': write Program to Count the Number of Each Vowel
#### write Program to Count the Number of Each Vowel ####
------------------------------------------------------------
def count_vowels(text):
	count=0
	for letter in text:
		if letter in vowels:
			count+=1
	return count
------------------------------------------------------------
Enter your question or type 'exit': write a python function to copy the sign bit
#### write a python function to copy the sign bit ####
------------------------------------------------------------
def <unk>(dst,src):
	return(dst)+(dst)
------------------------------------------------------------
Enter your question or type 'exit': write a python function to join directory names to create a path
#### write a python function to join directory names to create a path ####
------------------------------------------------------------
def <unk>(base_dir):
	file_path_args=0
	with open(filepath,'r')as f:
			f_read=f.read(f,f)
			return f_read
------------------------------------------------------------
Enter your question or type 'exit': write a python function to find linear interpolation between two points x and y given a variable t
#### write a python function to find linear interpolation between two points x and y given a variable t ####
------------------------------------------------------------
def <unk>(x1,y1,x2,y2,y2):
	return(((x1*(y1-y1))*(y2-y2))/(y2)
------------------------------------------------------------
Enter your question or type 'exit': write a python function to remove all digits and underscores from a Unicode strings
#### write a python function to remove all digits and underscores from a Unicode strings ####
------------------------------------------------------------
def clean_str(text):
	import re
	return re.sub(text)
------------------------------------------------------------
Enter your question or type 'exit': swap two numbers
#### swap two numbers ####
------------------------------------------------------------
def <unk>(num1,num2):
	mul=num1*num2
	return mul
------------------------------------------------------------
Enter your question or type 'exit': write a python function to simulate an exception and log the error using logger provided by the user.
#### write a python function to simulate an exception and log the error using logger provided by the user. ####
------------------------------------------------------------
def <unk>():
	import random
	try:
		print("division by")
	except valueerror:
		print(""""""")
		print("destination.glob.glob())
		except valueerror:
	print(")
	print("destination)
		print("destination.")
		print(")
------------------------------------------------------------
Enter your question or type 'exit': write a python program to subtract two numbers
#### write a python program to subtract two numbers ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'sum:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': write a python Program to calculate the square root
#### write a python Program to calculate the square root ####
------------------------------------------------------------
num=8
num_sqrt=num*0.5
print('the square root of%0.3f is%0.3f'%(num,num_sqrt))
------------------------------------------------------------
Enter your question or type 'exit': write a program to find and print the largest among three numbers
#### write a program to find and print the largest among three numbers ####
------------------------------------------------------------
num1=10
num2=12
num3=14
if(num1>=num2)and(num1>=num3):
	largest=num1
elif(num2>=num1)and(num2>=num3):
	largest=num2
else:
	largest=num3
print(f'largest:{largest}')
------------------------------------------------------------
Enter your question or type 'exit': write a python program to swap two variables
#### write a python program to swap two variables ####
------------------------------------------------------------
x=10
y=10
x,y=y,x
print('the value of y after swapping:{}'.format(x))
------------------------------------------------------------
Enter your question or type 'exit': write Program to Convert Kilometers to Miles
#### write Program to Convert Kilometers to Miles ####
------------------------------------------------------------
kilometers=float(input('enter the kilometers:'))
conv_fac=0.621371
miles=kilometers*conv_fac
print('%0.2f kilometers is equal to%0.2f miles'%(kilometers,miles))
------------------------------------------------------------
Enter your question or type 'exit': write Program to Convert Celsius To Fahrenheit                                              
#### write Program to Convert Celsius To Fahrenheit ####
------------------------------------------------------------
celsius=37.5
fahrenheit=(celsius*1.8)+32
print('%0.1f degree celsius is equal to%0.1f degree fahrenheit'%(celsius,fahrenheit))
------------------------------------------------------------
Enter your question or type 'exit': write Program to Check if a Number is Positive, Negative or 0
#### write Program to Check if a Number is Positive,Negative or 0 ####
------------------------------------------------------------
num=int(input("enter a number:"))
if num>0:
	print("positive number")
elif num==0:
	print("zero")
else:
	print("negative number")
------------------------------------------------------------
Enter your question or type 'exit': write a python Program to Check if a Number is Odd or Even
#### write a python Program to Check if a Number is Odd or Even ####
------------------------------------------------------------
num=int(input("enter a number:"))
if num>0:
	print("{0}is even".format(num))
else:
	print("{0}is odd".format(num))
------------------------------------------------------------
Enter your question or type 'exit': write Program to Check Leap Year
#### write Program to Check Leap Year ####
------------------------------------------------------------
year=2000
if(year%4)==0:
	if(year%100)==0:
		if(year%400)==0:
			print("{0}is a leap year".format(year))
			else:
			print("{0}is a leap year".format(year))
	else:
		print("{0}is not a leap year".format(year))
else:
	print("{0}is not a leap year".format(year))
------------------------------------------------------------
Enter your question or type 'exit': write a Python Program to check if a number is prime or not
#### write a Python Program to check if a number is prime or not ####
------------------------------------------------------------
def <unk>(num):
	if num>1:
		for i in range(2,num):
			if(num%i)==0:
				print(num,"is not a prime number")
				break
	else:
				print(num,"is not a prime number")
					print(num,"is not a prime number")
	else:
	print(num,"is not a prime number")
------------------------------------------------------------
Enter your question or type 'exit': write program to display all the prime numbers within an interval
#### write program to display all the prime numbers within an interval ####
------------------------------------------------------------
lower=900
upper=1000
for num in range(lower,upper+1):
	if num>1:
		for i in range(2,num):
			if(num%i)==0:
					break
			else:
				print(num)
------------------------------------------------------------
Enter your question or type 'exit': write Program to Display the multiplication Table
#### write Program to Display the multiplication Table ####
------------------------------------------------------------
num=int(input("please enter a number:"))
for i in range(1,11):
	print(num,'x',i,'=',num*i)
------------------------------------------------------------
Enter your question or type 'exit': write Program to Check Armstrong Number 
#### write Program to Check Armstrong Number ####
------------------------------------------------------------
num=int(input("enter a number:"))
sum=0
temp=num
while temp>0:
	digit=temp%10
	sum+=digit**3
	temp/=10
if num==sum:
	print(num,"is an armstrong number")
else:
	print(num,"is not an armstrong number")
------------------------------------------------------------
Enter your question or type 'exit': write a Python Program to Find the Sum of Natural Numbers
#### write a Python Program to Find the Sum of Natural Numbers ####
------------------------------------------------------------
def sumdigits(n):
	if n<=0:
		return n
	else:
		return n*(n-1)+recur_sum(n-2)
n=16
print("sum of 0 is",recur_sum(n))
------------------------------------------------------------
Enter your question or type 'exit': write Program to Find Numbers Divisible by Another Number
#### write Program to Find Numbers Divisible by Another Number ####
------------------------------------------------------------
def <unk>(n):
	if(n%2==0):
		return n
	else:
		return(n-1)
n=int(input("enter number:"))
print("number is odd number")
------------------------------------------------------------
Enter your question or type 'exit': write Program to Convert Decimal to Binary
#### write Program to Convert Decimal to Binary ####
------------------------------------------------------------
dec=344
print("the decimal value of",dec,"is:")
print(bin(dec),"in binary.")
------------------------------------------------------------
Enter your question or type 'exit': write Program to Convert Decimal to Hexadecimal
#### write Program to Convert Decimal to Hexadecimal ####
------------------------------------------------------------
dec=344
print("the decimal value of",dec,"is:")
print(hex(dec),"in hexadecimal.")
------------------------------------------------------------
Enter your question or type 'exit': write Program to Find ASCII Value of Character
#### write Program to Find ASCII Value of Character ####
------------------------------------------------------------
def print_ascii(char):
	print(ord(char))
------------------------------------------------------------
Enter your question or type 'exit': write Program to Find HCF       
#### write Program to Find HCF ####
------------------------------------------------------------
def compute_hcf(x,y):
	if x>y:
		smaller=y
	else:
		smaller=x
	for i in range(1,smaller+1):
		if((((x%i==0))and(y%i==0)):
			hcf=i
	return hcf
------------------------------------------------------------
Enter your question or type 'exit': write Program to Find LCM
#### write Program to Find LCM ####
------------------------------------------------------------
def compute_lcm(x,y):
	if x>y:
		greater=x
	else:
		greater=y
	while(true):
		if((greater%x==0)and(greater%y==0)):
			lcm=greater
			break
		greater+=1
	return lcm
------------------------------------------------------------
Enter your question or type 'exit': write a ptytho program to Count Tuple Elements Inside List
#### write a ptytho program to Count Tuple Elements Inside List ####
------------------------------------------------------------
def <unk>(*args):
	count=0
	for i in args:
		if i in test_list2:
		count+=1
		return count
------------------------------------------------------------
Enter your question or type 'exit': write a python program to Removes all items from the list
#### write a python program to Removes all items from the list ####
------------------------------------------------------------
my_list=[1,2,3,4,5,6,7,8,9,10]
print(my_list[-1])
------------------------------------------------------------
Enter your question or type 'exit': write a program to access first characters in a string
#### write a program to access first characters in a string ####
------------------------------------------------------------
word="hello world"
letter=word[0]
print(f"first charecter in string:{letter}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to access Last characters in a string
#### write a program to access Last characters in a string ####
------------------------------------------------------------
word="hello world"
letter=word[:-1]
print(f"first charecter in string:{letter}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Generate a list by list comprehension
#### write a program to Generate a list by list comprehension ####
------------------------------------------------------------
my_list=[1,2,3,4,5,6,7,8,9,10]
print(my_list[:5])
------------------------------------------------------------
Enter your question or type 'exit': write a program to Set the values in the new list to upper case
#### write a program to Set the values in the new list to upper case ####
------------------------------------------------------------
list1=[11,5,17,18,23,50]
unwanted_num={11,18}
for ele in list1:
	if ele%2==0:
		list1.remove(ele)
print("new list after removing all even numbers:",list1)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Sort the string list alphabetically
#### write a program to Sort the string list alphabetically ####
------------------------------------------------------------
thislist=["orange","mango","kiwi","pineapple","banana"]
thislist.sort()
print(f"sorted list:{thislist}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Join Two Sets
#### write a program to Join Two Sets ####
------------------------------------------------------------
a={1,2,3,4,5}
b={4,5,6,7,8}
print(a & b)
------------------------------------------------------------
Enter your question or type 'exit': write a program to keep only the items that are present in both sets
#### write a program to keep only the items that are present in both sets ####
------------------------------------------------------------
x={"apple","banana","cherry"}
y={"google","microsoft","apple"}
x.intersection_update(y)
print(f"duplicate value in two set:{x}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Create and print a dictionary
#### write a program to Create and print a dictionary ####
------------------------------------------------------------
thisdict={
	"brand":"ford",
	"model":"mustang",
	"year":1964
}
print(f"sample dictionary:{thisdict}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Evaluate a string and a number
#### write a program to Evaluate a string and a number ####
------------------------------------------------------------
print(bool("hello"))
------------------------------------------------------------
Enter your question or type 'exit': write a program to Calculate length of a string
#### write a program to Calculate length of a string ####
------------------------------------------------------------
word="hello world"
check=word.isdigit()
print(f"string ater replacement:{ksplit}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Count the number of spaces in a sring
#### write a program to Count the number of spaces in a sring ####
------------------------------------------------------------
def count_vowels(s):
	count=0
	for s in s:
		if s.count():
		count+=1
	return count
------------------------------------------------------------
Enter your question or type 'exit': write a program to Split Strings
#### write a program to Split Strings ####
------------------------------------------------------------
word="hello world"
ksplit=word.split('')
print(f"splited strings:{ksplit}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Prints ten dots
#### write a program to Prints ten dots ####
------------------------------------------------------------
ten="."*10
print(f"ten dots:{ten}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Replacing a string with another string
#### write a program to Replacing a string with another string ####
------------------------------------------------------------
str1="hello!it is a good thing"
substr1="good"
substr2="bad"
replaced_str=str1.replace(substr1,substr2)
print("string after replace:",str1)
------------------------------------------------------------
Enter your question or type 'exit': write a program to removes leading characters
#### write a program to removes leading characters ####
------------------------------------------------------------
word="xyz"
lstrip=word.lstrip()
print(f"string ater removal of leading characters:{lstrip}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to removes trailing characters
#### write a program to removes trailing characters ####
------------------------------------------------------------
word="xyz"
rstrip=word.rstrip()
print(f"string ater removal of trailing characters:{rstrip}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to check if all char are alphanumeric
#### write a program to check if all char are alphanumeric ####
------------------------------------------------------------
word="hello world"
check=word.isalnum()
print(f"string contains upper case?:{check}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Test if string starts with H
#### write a program to Test if string starts with H ####
------------------------------------------------------------
word="hello world"
check=word.startswith('h')
print(f"string starts with h?:{check}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Returns an integer value for the given character
#### write a program to Returns an integer value for the given character ####
------------------------------------------------------------
def print_ascii(char):
	return ord(char)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Fibonacci series up to 100
#### write a program to Fibonacci series up to 100 ####
------------------------------------------------------------
def recur_fibo(n):
	if n<=1:
		return n
	else:
		return n+fib(n-1)
n=int(input("enter second number:"))
nterms=int(input("enter second number:"))
print("fibonacci sequence:")
print("fibonacci sequence:"))
------------------------------------------------------------
Enter your question or type 'exit': write a program to Insert a number at the beginning of the queue
#### write a program to Insert a number at the beginning of the queue ####
------------------------------------------------------------
q=[1,2,3,4,5,6,7,8]
q.insert(0,5)
print(f"revised list:{q}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Addition of two vector
#### write a program to Addition of two vector ####
------------------------------------------------------------
v1=[1,2,3]
v2=[1,2,3]
s1=[0,0,0]
for i in range(len(v1)):
	s1[i]=v1[i]
print(f"new vector vector vector:{s1}")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Convert dictionary to JSON
#### write a program to Convert dictionary to JSON ####
------------------------------------------------------------
import json
person_dict={"name":"bob",
"languages":"english",
":"age":none,
":none,
"class":none,
}
with open("w")as json.json.json(person_dict,indent=true)as json_file:
	json.dump(person_dict,indent=true)
print("done writing json data into a file")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Remove empty strings from the list of strings
#### write a program to Remove empty strings from the list of strings ####
------------------------------------------------------------
list1=['mike','  ','  ','kelly',','emma']
print(list1[-1])
------------------------------------------------------------
Enter your question or type 'exit': write a program to Pick a random character from a given String
#### write a program to Pick a random character from a given String ####
------------------------------------------------------------
import random
name='pynative'
char=random.choice(name)
print("random char is",char)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Generate  random String of length 5
#### write a program to Generate   random String of length 5 ####
------------------------------------------------------------
import random
def randomstring(stringlength):
	return random.choice(letters)
------------------------------------------------------------
Enter your question or type 'exit': write a program, Given an input string, count occurrences of all characters within a string
#### write a program,Given an input string,count occurrences of all characters within a string ####
------------------------------------------------------------
str1="ababccd12@"
str2="bb123cca1@"
matched_chars=str1.count()
print("no.of matching characters are:"+str(len(matched_chars)))
------------------------------------------------------------
Enter your question or type 'exit': write a program to Reverse a given string
#### write a program to Reverse a given string ####
------------------------------------------------------------
str1="pynative"
countdict=dict()
for char in str1:
	char=str1[char]
	if char in countdict:
		countdict[char]=1
	else:
				countdict.append(char)
print('no')
------------------------------------------------------------
Enter your question or type 'exit': write a program for Removal all the characters other than integers from string
#### write a program for Removal all the characters other than integers from string ####
------------------------------------------------------------
str1="whatisthis"
split_string=list('.join(str1))
print(split_string)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Find the day of week of a given date?
#### write a program to Find the day of week of a given date? ####
------------------------------------------------------------
from datetime import datetime
given_date=datetime(2020,2,25)
days_to_subtract=7
res_date=given_date-timedelta(days=days_to_subtract)
print(res_date)
------------------------------------------------------------
Enter your question or type 'exit': write a recursive function to calculate the sum of numbers from 0 to 10
#### write a recursive function to calculate the sum of numbers from 0 to 10 ####
------------------------------------------------------------
def calculatesum(n):
	if n<=1:
		return n
	else:
		return n+recur_sum(n-1)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Generate a Python list of all the even numbers between two given numbers
#### write a program to Generate a Python list of all the even numbers between two given numbers ####
------------------------------------------------------------
list1=[1,2,3,4,5,6,7,8]
list2=[i for i in range(len(list1))if i%2==0]
print("new list after removing all even numbers:",list2)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Return the largest item from the given list
#### write a program to Return the largest item from the given list ####
------------------------------------------------------------
def <unk>(alist):
	for i in range(0,len(alist)):
		smallest=i
		while(alist[i]>alist[j]):
			alist[j]=alist[j]
			alist[j-1]=alist[j-1],alist[j+1]
alist=[2,3,4,5,6,5]
insertion_sort(alist)
print(alist)
------------------------------------------------------------
Enter your question or type 'exit': find largest number
#### find largest number ####
------------------------------------------------------------
def <unk>(n):
	if n<=0:
		return n
	else:
		return n+recur_sum(n-1)
n=int(input("enter number:"))
print("number is")
------------------------------------------------------------
Enter your question or type 'exit': write a program to Shuffle a list randomly
#### write a program to Shuffle a list randomly ####
------------------------------------------------------------
from random import shuffle
mylist=[3,6,7,8]
shuffle(mylist)
print(mylist)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Generate a random n-dimensional array of float numbers
#### write a program to Generate a random n-dimensional array of float numbers ####
------------------------------------------------------------
import random
print(random.sample(range(100),5))
------------------------------------------------------------
Enter your question or type 'exit': write a program to Generate random Universally unique IDs
#### write a program to Generate random Universally unique IDs ####
------------------------------------------------------------
import uuid
safeid=uuid.uuid4()
print("safe unique id is",safeid)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Choose given number of elements from the list with different probability
#### write a program to Choose given number of elements from the list with different probability ####
------------------------------------------------------------
import random
num1=5
numberlist=[111,222,333,444,555]
print(random.choices(numberlist,weights=(10,20,30,40,50,50,50,40,50,50),k=num1))
------------------------------------------------------------
Enter your question or type 'exit': write a program for generating a reliable secure random number
#### write a program for generating a reliable secure random number ####
------------------------------------------------------------
import random
print(random.sample())
------------------------------------------------------------
Enter your question or type 'exit': write a program to Calculate memory is being used by an list in Python
#### write a program to Calculate memory is being used by an list in Python ####
------------------------------------------------------------
import sys
list1=['scott','eric','eric','emma','smith']
print("size of list:",sys.getsizeof(list1))
------------------------------------------------------------
Enter your question or type 'exit': write a program to Find if all elements in a list are identical
#### write a program to Find if all elements in a list are identical ####
------------------------------------------------------------
list1=[11,-21,0,45,66,-93]
for num in list1:
	if num>=0:
		print(num,end="")
------------------------------------------------------------
Enter your question or type 'exit':  write a program to Merge two dictionaries in a single expression
####   write a program to Merge two dictionaries in a single expression ####
------------------------------------------------------------
currentemployee={1:'scott',2:"eric",3:"kelly"}
formeremployee  ={2:'eric',4:"emma"}
allemployee={*currentemployee,***formeremployee}
print(allemployee)
------------------------------------------------------------
Enter your question or type 'exit': write a program to Convert two lists into a dictionary
#### write a program to Convert two lists into a dictionary ####
------------------------------------------------------------
def map_dict(test_list1,test_list2,test_list3):
	res=[{a:{b:c}for(a,b,b,c)in zip(test_list1,test_list2,test_list3)]
	return res
------------------------------------------------------------
Enter your question or type 'exit': write a program to Alternate cases in String
#### write a program to Alternate cases in String ####
------------------------------------------------------------
test_str="geeksforgeeks"
print("the original string is:"+test_str)
all_freq={}
for i in test_str:
	if i in all_freq:
		all_freq[i]+=1
	else:
		all_freq[i]=1
res=min(all_freq,key=all_freq.get)
print("the maximum of all characters in geeksforgeeks is:"+str(res))
------------------------------------------------------------
Enter your question or type 'exit': write function to return the nth fibonacci number
#### write function to return the nth fibonacci number ####
------------------------------------------------------------
def fib(n):
	if n<=1:
		return n
	else:
		return n+fib(n-1)
------------------------------------------------------------
Enter your question or type 'exit': write function to return the factorial of a number
#### write function to return the factorial of a number ####
------------------------------------------------------------
def factorial(n):
	if n==1:
		return n
	else:
		return n*factorial(n-1)
------------------------------------------------------------
Enter your question or type 'exit': write function to return the squares of a list of numbers
#### write function to return the squares of a list of numbers ####
------------------------------------------------------------
def <unk>(nums):
	return[i*i for i in nums]
------------------------------------------------------------
Enter your question or type 'exit': write function to return the squareroot of a list of numbers
#### write function to return the squareroot of a list of numbers ####
------------------------------------------------------------
def <unk>(nums):
	return[i*i for i in nums]
------------------------------------------------------------
Enter your question or type 'exit': write function to add even number from 1st list and odd number from 2nd list
def even_odd(l1, l2):#### write function to add even number from 1st list and odd number from 2nd list ####
------------------------------------------------------------
def add_even_odd_list(l1:list,l2:list)->list:
	return[i+j for i,j in zip(l1,l2)if i%2==0]
------------------------------------------------------------
Enter your question or type 'exit': 
#### def even_odd(l1,l2 ): ####
------------------------------------------------------------
def <unk>(n):
	if n<=0:
		return 0
	elif n==1:
		return 1
	else:
			return 2**n+1
------------------------------------------------------------
Enter your question or type 'exit': write function to strip vowels from a string
#### write function to strip vowels from a string ####
------------------------------------------------------------
def <unk>(string):
	return string.replace('',')
------------------------------------------------------------
Enter your question or type 'exit': write ReLu function
#### write ReLu function ####
------------------------------------------------------------
def relu(x:list)->float:
	return x if x<0 else x
------------------------------------------------------------
Enter your question or type 'exit': write function to add even mubers in a list
#### write function to add even mubers in a list ####
------------------------------------------------------------
def add_even_num(l1,l2):
	return[i*j for i in l1 if i%2==0]
------------------------------------------------------------
Enter your question or type 'exit': write function to find the area of a circle
#### write function to find the area of a circle ####
------------------------------------------------------------
def findarea(r):
	pi=3.142
	return pi*r*r
------------------------------------------------------------
Enter your question or type 'exit': write function to return the cubes of a list of numbers
#### write function to return the cubes of a list of numbers ####
------------------------------------------------------------
def cube(l):
	return[i***i for i in range(l)]
------------------------------------------------------------
Enter your question or type 'exit': write function to create adders
#### write function to create adders ####
------------------------------------------------------------
def create_adders():
	adders=""
	def inner():
		for i in range(0,len()):
			for j in range(0,i):
					adders[j]="""""""
			for j in range(i):
						print(",",",end=""")
			print(")
		print(")
------------------------------------------------------------
Enter your question or type 'exit': write function for datetime
#### write function for datetime ####
------------------------------------------------------------
from datetime import datetime
date_string="feb 25 2020   4:20pm"
datetime_object=datetime.strptime(date_string,'%b%d%y%i:%m%p')
print(datetime_object)
------------------------------------------------------------
Enter your question or type 'exit': write program to compare strings
#### write program to compare strings ####
------------------------------------------------------------
word="hello world"
check=word.isalnum()
print(f"string contains digits?:{check}")
------------------------------------------------------------
Enter your question or type 'exit': write function for factorial using reduce
#### write function for factorial using reduce ####
------------------------------------------------------------
def factorial(n):
	if n==1:
		return n
	else:
		return n*factorial(n-1)
------------------------------------------------------------
Enter your question or type 'exit': write program to find if given co-ordinates are inside circle
#### write program to find if given co-ordinates are inside circle ####
------------------------------------------------------------
import math
radius=5
print(f'area:{math.pi*radius}')
------------------------------------------------------------
Enter your question or type 'exit': write function to find the area of a circle
#### write function to find the area of a circle ####
------------------------------------------------------------
def findarea(r):
	pi=3.142
	return pi*r*r
------------------------------------------------------------
Enter your question or type 'exit': write program for the sum of first n numbers.
#### write program for the sum of first n numbers. ####
------------------------------------------------------------
def sumdigits(n):
	if n<=0:
		return n
	else:
		return n+recur_sum(n-1)
n=16
print("sum of digits is",recur_sum(n))
------------------------------------------------------------
Enter your question or type 'exit': write Program to Concatenate Strings
#### write Program to Concatenate Strings ####
------------------------------------------------------------
str1="hello world"
str2=str1.replace('')
print('string is:',str1)
------------------------------------------------------------
Enter your question or type 'exit': write a Python function To Calculate Volume OF Cylinder
#### write a Python function To Calculate Volume OF Cylinder ####
------------------------------------------------------------
def cal_cylinder_volume(height,radius):
	pi=3.14
	return pi*radius*height*height
------------------------------------------------------------
Enter your question or type 'exit': write a program to Recursive Python function to solve the tower of hanoi
#### write a program to Recursive Python function to solve the tower of hanoi ####
------------------------------------------------------------
def towerofhanoi(n,source,destination,auxiliary):
	if n==1:
		print("move disk 1 from source",source"to destination",destination",destination)
		towerofhanoi(n-1,destination)
		print("move disk",",",n,"from source"to destination",source",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination",destination
------------------------------------------------------------
Enter your question or type 'exit': write function to find angle between hour hand and minute hand
#### write function to find angle between hour hand and minute hand ####
------------------------------------------------------------
import math
def <unk>(angle):
	return math.sin(angle)
------------------------------------------------------------
Enter your question or type 'exit': write function for finding cosine angle
#### write function for finding cosine angle ####
------------------------------------------------------------
def cosine(angle):
	return(angle)//(angle)
------------------------------------------------------------
Enter your question or type 'exit': write function for finding the derivative of sine angle
#### write function for finding the derivative of sine angle ####
------------------------------------------------------------
import math
def sin(angle):
	return math.cos(angle)
------------------------------------------------------------
Enter your question or type 'exit': write function for finding the derivative of tangent angle
#### write function for finding the derivative of tangent angle ####
------------------------------------------------------------
import math
def <unk>(angle):
	return math.cos(angle)
------------------------------------------------------------
Enter your question or type 'exit': write function for finding the exponent of a number
#### write function for finding the exponent of a number ####
------------------------------------------------------------
def <unk>(x):
	return x**2
------------------------------------------------------------
Enter your question or type 'exit': write a program to increment number which is at end of string
#### write a program to increment number which is at end of string ####
------------------------------------------------------------
def extract_alpha(text):
	for c in text:
		if c.isalpha():
				d[:]+=1
		else:
				c.append(c)
	return c
------------------------------------------------------------
Enter your question or type 'exit': write a function to check if a lower case letter exists in a given string
#### write a function to check if a lower case letter exists in a given string ####
------------------------------------------------------------
def check_lower(str1):
	for char in str1:
		k=char.islower()
		if k==true:
			return true
	return false
------------------------------------------------------------
Enter your question or type 'exit': write a program to print number of words in a string
#### write a program to print number of words in a string ####
------------------------------------------------------------
str1="it is wonderful and sunny day for a picnic in the park"
str_len="
res_str=[]
text=str1.split(""")
for char in text:
	if len(word)>=":
		res_str.append(word)
print("number of words that are:"+str(res_str))
------------------------------------------------------------
Enter your question or type 'exit': write a program to print number of characters in a string
#### write a program to print number of characters in a string ####
------------------------------------------------------------
str1="pynative"
digitcount=0
for i in range(0,len(str1)):
	char=str1[i]
	if(char.isdigit()):
		digitcount+=1
print('number of digits:',digitcount)
------------------------------------------------------------
Enter your question or type 'exit': write a funtion that accepts two lists of equal length and converts them into a dictioinary
#### write a funtion that accepts two lists of equal length and converts them into a dictioinary ####
------------------------------------------------------------
def list_to_dict(list1,list2):
	return dict(zip(list1,list2))
------------------------------------------------------------
Enter your question or type 'exit': write a python function that accepts a list of dictionaries and sorts it by a specified key
#### write a python function that accepts a list of dictionaries and sorts it by a specified key ####
------------------------------------------------------------
def sort_dict_list(dict_list,sort_key):
	dict_list.sort(key=lambda item:item.get(sort_key))
------------------------------------------------------------
Enter your question or type 'exit': write a program to capitalize the first and last character of each key in a dictionary
#### write a program to capitalize the first and last character of each key in a dictionary ####
------------------------------------------------------------
def invert_dict(input_dict):
	inverted_dict={key:key for key,val in input_dict.items()}
	return my_inverted_dict
------------------------------------------------------------
Enter your question or type 'exit': write a python function that accepts a dictionary that has unique values and returns its inversion
#### write a python function that accepts a dictionary that has unique values and returns its inversion ####
------------------------------------------------------------
def invert_dict(input_dict):
	my_inverted_dict={value:key for key,value in input_dict.items()}
	return my_inverted_dict
------------------------------------------------------------
Enter your question or type 'exit': sum of two number
#### sum of two number ####
------------------------------------------------------------
def add_two_numbers(num1,num2):
	sum=num1+num2
	return sum
------------------------------------------------------------
Enter your question or type 'exit': write a python function to return a flattened dictionary from a nested dictionary input
#### write a python function to return a flattened dictionary from a nested dictionary input ####
------------------------------------------------------------
def flatten_dict(dd,separator='_',prefix='):
	flattened= {prefix+separator+separator+k if prefix else k:v for kk,vv in dd.items()for k,v in flatten_dict(vv,separator,kk)}if isinstance(dd,dict)else{prefix:dd}
	return flattened
------------------------------------------------------------
Enter your question or type 'exit': write a program to combine two dictionaries using a priority dictionary and print the new combined dictionary.
#### write a program to combine two dictionaries using a priority dictionary and print the new combined dictionary. ####
------------------------------------------------------------
def merge(dict1,dict2):
	return{*dict1,*dict2}
------------------------------------------------------------
Enter your question or type 'exit': write a Python program that sorts dictionary keys to a list using their values and prints this list.
#### write a Python program that sorts dictionary keys to a list using their values and prints this list. ####
------------------------------------------------------------
test_dict={'gfg':1,'is':2,'best':3}
res={key:val for key,val in test_dict.items()}
for key,val in test_dict.items():
	res.setdefault()
	res.setdefault(val)
print("the dictionary after conversion:"+str(res))
------------------------------------------------------------
Enter your question or type 'exit': write a python function to add elements of two lists
#### write a python function to add elements of two lists ####
------------------------------------------------------------
def append_lists(l1:list,l2:list)->list:
	return l1.extend(l2)
------------------------------------------------------------
Enter your question or type 'exit': write a  python program to print the last element of a list
#### write a   python program to print the last element of a list ####
------------------------------------------------------------
list1=[11,5,17,18,23,50]
unwanted_num={11,18}
for item in list1:
	if item not in unwanted_num:
		count[item]=count+1
print("printing",list1)
------------------------------------------------------------
Enter your question or type 'exit': write a python fuction to create an empty list
#### write a python fuction to create an empty list ####
------------------------------------------------------------
def emptylist():
	return list()
------------------------------------------------------------
Enter your question or type 'exit': write a python program to print a list with all elements as 5 and of length 10
#### write a python program to print a list with all elements as 5 and of length 10 ####
------------------------------------------------------------
list1=[11,5,17,18,23,50]
unwanted_num=[]
for ele in list1:
	if ele not in unwanted_num:
		list1.append(ele)
print("new list after removing unwanted numbers:",list1)
------------------------------------------------------------
Enter your question or type 'exit': write a python program to reverse a list and print it.
#### write a python program to reverse a list and print it. ####
------------------------------------------------------------
my_list=[1,2,3,4,5,6,7,8,9,10]
print(my_list[:])
------------------------------------------------------------
Enter your question or type 'exit': write a python program to swap first and last element of a list . Print the final list
#### write a python program to swap first and last element of a list.Print the final list ####
------------------------------------------------------------
list1=[11,5,17,18,23,50]
unwanted_num={11,18}
list1=[ele for ele in list1 if ele not in unwanted_num]
print("new list after removing unwanted numbers:",list1)
------------------------------------------------------------
Enter your question or type 'exit': write a python program for print all elements with digit 7.
#### write a python program for print all elements with digit 7. ####
------------------------------------------------------------
test_list=[56,72,875,173]
k=7
res=[ele for ele in test_list if str(ele)in str(ele)]
print("elements with digit k:"+str(res))
------------------------------------------------------------
Enter your question or type 'exit': write a python program that would print the first n positive integers using a for loop
#### write a python program that would print the first n positive integers using a for loop ####
------------------------------------------------------------
def reverse_integer(n):
	if n<=0:
		return n
	else:
		return n+recursive_sum(n-1)
------------------------------------------------------------
Enter your question or type 'exit': input list sorted in descending order
#### input list sorted in descending order ####
------------------------------------------------------------
def sort_descending(list_to_be_sorted):
	return sorted(list_to_be_sorted,reverse=true)
------------------------------------------------------------
Enter your question or type 'exit': sum of first n natural numbers, where n is the input
#### sum of first n natural numbers,where n is the input ####
------------------------------------------------------------
def <unk>(n):
	if n<=0:
		return n
	else:
		return n+sum_of_nums(n-1)
------------------------------------------------------------
Enter your question or type 'exit': square of a given input number
#### square of a given input number ####
------------------------------------------------------------
def square(x):
	return x**2
------------------------------------------------------------
Enter your question or type 'exit': python program that asks for user input and prints the given input
#### python program that asks for user input and prints the given input ####
------------------------------------------------------------
a=input("user input")
print(a)
------------------------------------------------------------
Enter your question or type 'exit': program that would merge two dictionaries by adding the second one into the first
#### program that would merge two dictionaries by adding the second one into the first ####
------------------------------------------------------------
a={"a":1,"b":2,"c":3}
b={"c":4,"d":3}
a.update(b)
------------------------------------------------------------
Enter your question or type 'exit': python function that would reverse the given string
#### python function that would reverse the given string ####
------------------------------------------------------------
def reverse_string(str_to_be_reversed):
	return str_to_be_reversed[:::-1]
------------------------------------------------------------
Enter your question or type 'exit': write a python program that would print "Hello World"
#### write a python program that would print"Hello World" ####
------------------------------------------------------------
unicodestring=u"hello world!"
print(unicodestring)
------------------------------------------------------------
Enter your question or type 'exit': write a python program that would swap variable values
#### write a python program that would swap variable values ####
------------------------------------------------------------
x=10
y=10
x=y
print('the value of x after swapping:{}'.format(x))
------------------------------------------------------------
Enter your question or type 'exit': swap values
#### swap values ####
------------------------------------------------------------
x={'a':1,'b':2}
y={'a':1,'b':2}
print("the <unk> of y:")
print("after swapping:")
print("after swapping:")
print("after swapping:",x)
------------------------------------------------------------
Enter your question or type 'exit': write a python program that iterates over a dictionary and prints its keys and values
#### write a python program that iterates over a dictionary and prints its keys and values ####
------------------------------------------------------------
thisdict={
	"brand":"ford",
	"model":"mustang",
	"year":1964
}
print(f"sample dictionary:{thisdict}")
------------------------------------------------------------
Enter your question or type 'exit': deletes the last element of a list 
#### deletes the last element of a list ####
------------------------------------------------------------
def delete_last_element(list_to_be_processed):
	deleted_element=list_to_be_processed.pop()
	return list_to_be_processed,deleted_element
------------------------------------------------------------
Enter your question or type 'exit': write a python function to that performs as ReLU
#### write a python function to that performs as ReLU ####
------------------------------------------------------------
def relu(x:float)->float:
	import math
	return math.exp(x)
------------------------------------------------------------
Enter your question or type 'exit': swap two numbers and print them
#### swap two numbers and print them ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'sum:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': python function to get the maximum element in a list
#### python function to get the maximum element in a list ####
------------------------------------------------------------
def swaplist(lst):
	return lst.count(lst)
------------------------------------------------------------
Enter your question or type 'exit': write a python program to tokenise a string into words and print them
#### write a python program to tokenise a string into words and print them ####
------------------------------------------------------------
str1="it is wonderful and sunny day for a picnic in the park"
str_len="
res_str=[]
text=str1.split("")
for i in text:
	if(i.isdigit()):
		count+=1
print("the number of words that are:"+str(res))
------------------------------------------------------------
Enter your question or type 'exit': write a python program to print the command line arguements given to a file
#### write a python program to print the command line arguements given to a file ####
------------------------------------------------------------
import sys
print("python keywords:")
print(sys.kwlist)
------------------------------------------------------------
Enter your question or type 'exit': write a python program to print a string in lowercase
#### write a python program to print a string in lowercase ####
------------------------------------------------------------
str1="pynative"
digitcount=0
for i in range(0,len(str1)):
	char=str1[i]
	if(char.isdigit()):
		digitcount+=1
print('number of upper case:',digitcount)
------------------------------------------------------------
Enter your question or type 'exit': terminate the program execution
#### terminate the program execution ####
------------------------------------------------------------
import sys
print(sys.version)
------------------------------------------------------------
Enter your question or type 'exit': program to print the datatype of a variable
#### program to print the datatype of a variable ####
------------------------------------------------------------
x=10
y=10
print(f'the value of x:{x}')
------------------------------------------------------------
Enter your question or type 'exit': exit
