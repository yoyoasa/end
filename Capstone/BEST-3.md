# BEST-3

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

LEARNING_RATE = 0.0001
MAX_LR = 0.00095
N_EPOCHS = 35
CLIP = 1
STEPS_PER_EPOCH = len(train_iterator)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# One Cycle Scheduler
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS, anneal_strategy='linear')

# One cycle schedule with custome function
schedule = np.interp(np.arange(N_EPOCHS+1), [0, 5, 20, N_EPOCHS], [LEARNING_RATE, MAX_LR, LEARNING_RATE/5.0, LEARNING_RATE/10.0])
def lr_schedules(epoch):
    return schedule[epoch+1]


Device: cuda
Data Set Size 4136
Valid Data Size 413
Unique tokens in Question vocabulary: 1376
Unique tokens in Answer vocabulary: 3103
The model has 9,178,199 trainable parameters
Pretrained embedding dimension torch.Size([3103, 300])
Learning Rate: 0.00027
Epoch: 01 | Time: 0m 8s
	Train Loss: 4.629 | Train PPL: 102.434
	 Val. Loss: 3.468 |  Val. PPL:  32.075
Learning Rate: 0.00043999999999999996
Epoch: 02 | Time: 0m 8s
	Train Loss: 3.281 | Train PPL:  26.607
	 Val. Loss: 2.838 |  Val. PPL:  17.090
Learning Rate: 0.00061
Epoch: 03 | Time: 0m 8s
	Train Loss: 2.810 | Train PPL:  16.606
	 Val. Loss: 2.488 |  Val. PPL:  12.042
Learning Rate: 0.00078
Epoch: 04 | Time: 0m 8s
	Train Loss: 2.516 | Train PPL:  12.375
	 Val. Loss: 2.282 |  Val. PPL:   9.792
Learning Rate: 0.00095
Epoch: 05 | Time: 0m 8s
	Train Loss: 2.310 | Train PPL:  10.074
	 Val. Loss: 2.138 |  Val. PPL:   8.486
Learning Rate: 0.000888
Epoch: 06 | Time: 0m 8s
	Train Loss: 2.111 | Train PPL:   8.258
	 Val. Loss: 1.966 |  Val. PPL:   7.140
Learning Rate: 0.000826
Epoch: 07 | Time: 0m 8s
	Train Loss: 1.909 | Train PPL:   6.743
	 Val. Loss: 1.853 |  Val. PPL:   6.376
Learning Rate: 0.0007639999999999999
Epoch: 08 | Time: 0m 8s
	Train Loss: 1.746 | Train PPL:   5.734
	 Val. Loss: 1.724 |  Val. PPL:   5.609
Learning Rate: 0.0007019999999999999
Epoch: 09 | Time: 0m 8s
	Train Loss: 1.598 | Train PPL:   4.941
	 Val. Loss: 1.638 |  Val. PPL:   5.145
Learning Rate: 0.0006399999999999999
Epoch: 10 | Time: 0m 8s
	Train Loss: 1.472 | Train PPL:   4.359
	 Val. Loss: 1.556 |  Val. PPL:   4.741
Learning Rate: 0.000578
Epoch: 11 | Time: 0m 8s
	Train Loss: 1.359 | Train PPL:   3.891
	 Val. Loss: 1.483 |  Val. PPL:   4.406
Learning Rate: 0.000516
Epoch: 12 | Time: 0m 7s
	Train Loss: 1.252 | Train PPL:   3.496
	 Val. Loss: 1.421 |  Val. PPL:   4.140
Learning Rate: 0.000454
Epoch: 13 | Time: 0m 7s
	Train Loss: 1.175 | Train PPL:   3.240
	 Val. Loss: 1.381 |  Val. PPL:   3.977
Learning Rate: 0.000392
Epoch: 14 | Time: 0m 7s
	Train Loss: 1.092 | Train PPL:   2.982
	 Val. Loss: 1.317 |  Val. PPL:   3.734
Learning Rate: 0.00033
Epoch: 15 | Time: 0m 7s
	Train Loss: 1.020 | Train PPL:   2.773
	 Val. Loss: 1.294 |  Val. PPL:   3.647
Learning Rate: 0.000268
Epoch: 16 | Time: 0m 7s
	Train Loss: 0.957 | Train PPL:   2.605
	 Val. Loss: 1.255 |  Val. PPL:   3.508
Learning Rate: 0.0002059999999999999
Epoch: 17 | Time: 0m 7s
	Train Loss: 0.903 | Train PPL:   2.466
	 Val. Loss: 1.220 |  Val. PPL:   3.386
Learning Rate: 0.00014399999999999992
Epoch: 18 | Time: 0m 7s
	Train Loss: 0.858 | Train PPL:   2.358
	 Val. Loss: 1.198 |  Val. PPL:   3.315
Learning Rate: 8.199999999999993e-05
Epoch: 19 | Time: 0m 7s
	Train Loss: 0.819 | Train PPL:   2.267
	 Val. Loss: 1.175 |  Val. PPL:   3.238
Learning Rate: 2e-05
Epoch: 20 | Time: 0m 7s
	Train Loss: 0.793 | Train PPL:   2.210
	 Val. Loss: 1.168 |  Val. PPL:   3.216
Learning Rate: 1.9333333333333333e-05
Epoch: 21 | Time: 0m 7s
	Train Loss: 0.784 | Train PPL:   2.191
	 Val. Loss: 1.165 |  Val. PPL:   3.205
Learning Rate: 1.866666666666667e-05
Epoch: 22 | Time: 0m 7s
	Train Loss: 0.787 | Train PPL:   2.196
	 Val. Loss: 1.163 |  Val. PPL:   3.198
Learning Rate: 1.8e-05
Epoch: 23 | Time: 0m 7s
	Train Loss: 0.776 | Train PPL:   2.174
	 Val. Loss: 1.160 |  Val. PPL:   3.189
Learning Rate: 1.7333333333333336e-05
Epoch: 24 | Time: 0m 7s
	Train Loss: 0.772 | Train PPL:   2.164
	 Val. Loss: 1.157 |  Val. PPL:   3.181
Learning Rate: 1.6666666666666667e-05
Epoch: 25 | Time: 0m 8s
	Train Loss: 0.770 | Train PPL:   2.160
	 Val. Loss: 1.159 |  Val. PPL:   3.187
Learning Rate: 1.6000000000000003e-05
Epoch: 26 | Time: 0m 8s
	Train Loss: 0.768 | Train PPL:   2.155
	 Val. Loss: 1.156 |  Val. PPL:   3.178
Learning Rate: 1.5333333333333334e-05
Epoch: 27 | Time: 0m 8s
	Train Loss: 0.762 | Train PPL:   2.143
	 Val. Loss: 1.153 |  Val. PPL:   3.167
Learning Rate: 1.4666666666666668e-05
Epoch: 28 | Time: 0m 8s
	Train Loss: 0.763 | Train PPL:   2.145
	 Val. Loss: 1.154 |  Val. PPL:   3.170
Learning Rate: 1.4000000000000001e-05
Epoch: 29 | Time: 0m 8s
	Train Loss: 0.758 | Train PPL:   2.134
	 Val. Loss: 1.152 |  Val. PPL:   3.163
Learning Rate: 1.3333333333333335e-05
Epoch: 30 | Time: 0m 7s
	Train Loss: 0.756 | Train PPL:   2.130
	 Val. Loss: 1.150 |  Val. PPL:   3.159
Learning Rate: 1.2666666666666668e-05
Epoch: 31 | Time: 0m 7s
	Train Loss: 0.752 | Train PPL:   2.121
	 Val. Loss: 1.149 |  Val. PPL:   3.155
Learning Rate: 1.2e-05
Epoch: 32 | Time: 0m 7s
	Train Loss: 0.749 | Train PPL:   2.116
	 Val. Loss: 1.149 |  Val. PPL:   3.156
Learning Rate: 1.1333333333333334e-05
Epoch: 33 | Time: 0m 7s
	Train Loss: 0.746 | Train PPL:   2.110
	 Val. Loss: 1.148 |  Val. PPL:   3.151
Learning Rate: 1.0666666666666667e-05
Epoch: 34 | Time: 0m 7s
	Train Loss: 0.745 | Train PPL:   2.106
	 Val. Loss: 1.146 |  Val. PPL:   3.146
Learning Rate: 1e-05
Epoch: 35 | Time: 0m 7s
	Train Loss: 0.746 | Train PPL:   2.108
	 Val. Loss: 1.145 |  Val. PPL:   3.142
Best Model: capstone-model-34-1.1449779730576735.pt
Enter your question or type 'exit': addition
#### addition ####
------------------------------------------------------------
def add_even_odd_list(s1,s2):
	if len(s1)<len(s1):
		return
	elif s1[0]==1:
		return'length
	else:
		return'.join(s1[1:2,3])
------------------------------------------------------------
Enter your question or type 'exit': add two number
#### add two number ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'sum:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': multiply wto number
#### multiply wto number ####
------------------------------------------------------------
def <unk>(n):
	return n*(n*n*n*n)
------------------------------------------------------------
Enter your question or type 'exit': multiply two number
#### multiply two number ####
------------------------------------------------------------
def multiply(num1,num2):
	sum=num1*num2
	return sum
------------------------------------------------------------
Enter your question or type 'exit': substract two number
#### substract two number ####
------------------------------------------------------------
def <unk>(num1,num2):
	return num1%num2
------------------------------------------------------------
Enter your question or type 'exit': substring of a string
#### substring of a string ####
------------------------------------------------------------
from re import string
str1=''''
print("string starts string is",str1)
str1=str1.replace(str1)
print("string after <unk> string:",str1)
------------------------------------------------------------
Enter your question or type 'exit': replace string
#### replace string ####
------------------------------------------------------------
str1="hello world"
print("the original string is:"+str1)
str1=str1.replace(""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------
Enter your question or type 'exit': replace string from a substring
#### replace string from a substring ####
------------------------------------------------------------
from string import string
str1="/*jon is @developer & musician"
new_str=str1.translate(str1)
print("string after punctuation",str1)
------------------------------------------------------------
Enter your question or type 'exit': substring of a string
#### substring of a string ####
------------------------------------------------------------
from re import string
str1=''''
print("string starts string is",str1)
str1=str1.replace(str1)
print("string after <unk> string:",str1)
------------------------------------------------------------
Enter your question or type 'exit': fibonaccci
#### fibonaccci ####
------------------------------------------------------------
def <unk>():
	print('<unk>')
	print('<unk>')
	print('<unk>')
	print('<unk>')
	print('<unk>')
	print(',',',')
		print(',',',',',',',',',',',',',',',',',',',',',',',')
								print(',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',',','
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
print("plese plese plese plese plese enter a positive integer")
else:
	print(recur_fibo(nterms,","))
------------------------------------------------------------
Enter your question or type 'exit': print a fibonacci series
#### print a fibonacci series ####
------------------------------------------------------------
nterms=int(input("please enter a number:"))
n1,n2=0
if nterms<=0:
	print("please enter a positive integer")
elif nterms==1:
	print("fibonacci sequence:")
	print(n1)
	elif nterms===1:
	print("fibonacci sequence:")
	print(n1)
	print(n1)
	elif nterms=====0:
	print(":
	print(",",",",",",",",")
	print(n1)
	print(n1)
	print(n1)
	else:
	print(n1)
	print(n1)
	print("fibonacci sequence:")
	print(n1)
	print(n1)
	print(n1)
------------------------------------------------------------
Enter your question or type 'exit': exit
