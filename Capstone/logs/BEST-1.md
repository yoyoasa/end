## BEST-1

max_length = 150
max_decoder_length = max_length+50

BATCH_SIZE = 32

INPUT_DIM = len(Question.vocab)
OUTPUT_DIM = len(Answer.vocab)
HID_DIM = 300
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 5
DEC_HEADS = 5
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

LEARNING_RATE = 0.0005
MAX_LR = 0.001
N_EPOCHS = 30
CLIP = 1
STEPS_PER_EPOCH = len(train_iterator)

Device: cuda
Data Set Size 4135
Valid Data Size 413
Unique tokens in Question vocabulary: 1382
Unique tokens in Answer vocabulary: 3152
The model has 9,209,448 trainable parameters
Pretrained embedding dimension torch.Size([3152, 300])
Learning Rate: 4e-05
Epoch: 01 | Time: 0m 9s
	Train Loss: 5.874 | Train PPL: 355.813
	 Val. Loss: 4.351 |  Val. PPL:  77.592
Learning Rate: 0.0001467680608365019
Epoch: 02 | Time: 0m 9s
	Train Loss: 3.813 | Train PPL:  45.288
	 Val. Loss: 3.344 |  Val. PPL:  28.327
Learning Rate: 0.0002535361216730038
Epoch: 03 | Time: 0m 9s
	Train Loss: 3.163 | Train PPL:  23.647
	 Val. Loss: 2.948 |  Val. PPL:  19.061
Learning Rate: 0.0003603041825095057
Epoch: 04 | Time: 0m 9s
	Train Loss: 2.781 | Train PPL:  16.128
	 Val. Loss: 2.629 |  Val. PPL:  13.857
Learning Rate: 0.0004670722433460076
Epoch: 05 | Time: 0m 9s
	Train Loss: 2.489 | Train PPL:  12.043
	 Val. Loss: 2.431 |  Val. PPL:  11.373
Learning Rate: 0.0005738403041825095
Epoch: 06 | Time: 0m 9s
	Train Loss: 2.275 | Train PPL:   9.733
	 Val. Loss: 2.290 |  Val. PPL:   9.878
Learning Rate: 0.0006806083650190114
Epoch: 07 | Time: 0m 8s
	Train Loss: 2.115 | Train PPL:   8.289
	 Val. Loss: 2.180 |  Val. PPL:   8.844
Learning Rate: 0.0007873764258555133
Epoch: 08 | Time: 0m 9s
	Train Loss: 1.972 | Train PPL:   7.185
	 Val. Loss: 2.080 |  Val. PPL:   8.007
Learning Rate: 0.0008941444866920152
Epoch: 09 | Time: 0m 9s
	Train Loss: 1.855 | Train PPL:   6.389
	 Val. Loss: 1.964 |  Val. PPL:   7.130
Learning Rate: 0.0009995930012210012
Epoch: 10 | Time: 0m 8s
	Train Loss: 1.714 | Train PPL:   5.551
	 Val. Loss: 1.877 |  Val. PPL:   6.537
Learning Rate: 0.0009519741440781441
Epoch: 11 | Time: 0m 9s
	Train Loss: 1.578 | Train PPL:   4.847
	 Val. Loss: 1.805 |  Val. PPL:   6.082
Learning Rate: 0.000904355286935287
Epoch: 12 | Time: 0m 8s
	Train Loss: 1.445 | Train PPL:   4.240
	 Val. Loss: 1.725 |  Val. PPL:   5.612
Learning Rate: 0.0008567364297924298
Epoch: 13 | Time: 0m 8s
	Train Loss: 1.328 | Train PPL:   3.772
	 Val. Loss: 1.691 |  Val. PPL:   5.423
Learning Rate: 0.0008091175726495727
Epoch: 14 | Time: 0m 8s
	Train Loss: 1.224 | Train PPL:   3.401
	 Val. Loss: 1.652 |  Val. PPL:   5.216
Learning Rate: 0.0007614987155067155
Epoch: 15 | Time: 0m 8s
	Train Loss: 1.130 | Train PPL:   3.095
	 Val. Loss: 1.549 |  Val. PPL:   4.706
Learning Rate: 0.0007138798583638585
Epoch: 16 | Time: 0m 8s
	Train Loss: 1.054 | Train PPL:   2.869
	 Val. Loss: 1.544 |  Val. PPL:   4.682
Learning Rate: 0.0006662610012210012
Epoch: 17 | Time: 0m 9s
	Train Loss: 0.977 | Train PPL:   2.656
	 Val. Loss: 1.499 |  Val. PPL:   4.478
Learning Rate: 0.0006186421440781441
Epoch: 18 | Time: 0m 8s
	Train Loss: 0.912 | Train PPL:   2.490
	 Val. Loss: 1.447 |  Val. PPL:   4.249
Learning Rate: 0.000571023286935287
Epoch: 19 | Time: 0m 9s
	Train Loss: 0.850 | Train PPL:   2.340
	 Val. Loss: 1.475 |  Val. PPL:   4.372
Learning Rate: 0.0005234044297924298
Epoch: 20 | Time: 0m 8s
	Train Loss: 0.803 | Train PPL:   2.232
	 Val. Loss: 1.452 |  Val. PPL:   4.273
Learning Rate: 0.00047578557264957267
Epoch: 21 | Time: 0m 8s
	Train Loss: 0.749 | Train PPL:   2.114
	 Val. Loss: 1.436 |  Val. PPL:   4.203
Learning Rate: 0.0004281667155067155
Epoch: 22 | Time: 0m 8s
	Train Loss: 0.699 | Train PPL:   2.011
	 Val. Loss: 1.425 |  Val. PPL:   4.156
Learning Rate: 0.00038054785836385847
Epoch: 23 | Time: 0m 9s
	Train Loss: 0.655 | Train PPL:   1.926
	 Val. Loss: 1.403 |  Val. PPL:   4.066
Learning Rate: 0.0003329290012210012
Epoch: 24 | Time: 0m 9s
	Train Loss: 0.614 | Train PPL:   1.847
	 Val. Loss: 1.415 |  Val. PPL:   4.114
Learning Rate: 0.00028531014407814415
Epoch: 25 | Time: 0m 9s
	Train Loss: 0.580 | Train PPL:   1.785
	 Val. Loss: 1.399 |  Val. PPL:   4.052
Learning Rate: 0.000237691286935287
Epoch: 26 | Time: 0m 9s
	Train Loss: 0.542 | Train PPL:   1.719
	 Val. Loss: 1.390 |  Val. PPL:   4.017
Learning Rate: 0.00019007242979242984
Epoch: 27 | Time: 0m 8s
	Train Loss: 0.515 | Train PPL:   1.674
	 Val. Loss: 1.399 |  Val. PPL:   4.052
Learning Rate: 0.00014245357264957269
Epoch: 28 | Time: 0m 9s
	Train Loss: 0.482 | Train PPL:   1.620
	 Val. Loss: 1.401 |  Val. PPL:   4.057
Learning Rate: 9.483471550671564e-05
Epoch: 29 | Time: 0m 8s
	Train Loss: 0.464 | Train PPL:   1.590
	 Val. Loss: 1.385 |  Val. PPL:   3.994
Learning Rate: 4.721585836385837e-05
Epoch: 30 | Time: 0m 9s
	Train Loss: 0.447 | Train PPL:   1.564
	 Val. Loss: 1.385 |  Val. PPL:   3.997
Enter your question or type 'exit': sum
#### sum ####
------------------------------------------------------------
def sumdigits(num):
	sum=0
	while(num>0):
		sum+=num
		num-=1
	print(f'the sum is{sum}')
------------------------------------------------------------
Enter your question or type 'exit': addition
#### addition ####
------------------------------------------------------------
v1=[1,2,3]
v2=[1,2,3]
s1=[0,0]
for i in range(len(v1)):
	s1[i]=v1[i]+v2[i]
print(f"new vector:{s1}")
------------------------------------------------------------
Enter your question or type 'exit': substraction
#### substraction ####
------------------------------------------------------------
def <unk>(n):
	a=[0]*n
	a[1]=a[2,3,5,6]
	for i in range(n):
		a[i]=a[i]*(a[i]+a[i]+b)
		if a>=0:
			a[i]=a[i]=a[i]
			a[i]=a[i]
			a.append(a)
	return a.<unk>
print(a)
------------------------------------------------------------
Enter your question or type 'exit': fibonacci series
#### fibonacci series ####
------------------------------------------------------------
def recur_fibo(n):
	if n<=1:
		return n
	else:
		return(recur_fibo(n-1)+recur_fibo(n-2))
nterms=10
if nterms<=0:
	print("plese enter a positive integer")
else:
	print("fibonacci sequence:")
------------------------------------------------------------
Enter your question or type 'exit': fibonacci sequesne
#### fibonacci sequesne ####
------------------------------------------------------------
def recur_fibo(n):
	if n<=1:
		return n
	else:
		return(recur_fibo(n-1)+recur_fibo(n-2))
nterms=1
if nterms<=2:
	print("plese enter a positive integer")
else:
	print("fibonacci sequence:")
------------------------------------------------------------
Enter your question or type 'exit': factorial
#### factorial ####
------------------------------------------------------------
def factorial(n):
	if n==1:
		return n
	else:
		return n*factorial(n-1)
num=7
if num<0:
	print("sorry,factorial does not exist for negative numbers")
else:
	print("the factorial of 0 is 1",factorial)
------------------------------------------------------------
Enter your question or type 'exit': exit
