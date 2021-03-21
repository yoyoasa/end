# Best - 5 with Conala Dataset

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
N_EPOCHS = 24
CLIP = 1
STEPS_PER_EPOCH = len(train_iterator)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# One Cycle Scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS, anneal_strategy='linear')

Device: cuda
Data Set Size 7015
Valid Data Size 701
Unique tokens in Question vocabulary: 1378
Unique tokens in Answer vocabulary: 3100
The model has 9,176,996 trainable parameters
Pretrained embedding dimension torch.Size([3100, 300])
Learning Rate: 4e-05
Epoch: 01 | Time: 0m 12s
	Train Loss: 4.763 | Train PPL: 117.127
	 Val. Loss: 2.764 |  Val. PPL:  15.868
Learning Rate: 0.00017342692685666153
Epoch: 02 | Time: 0m 11s
	Train Loss: 2.597 | Train PPL:  13.422
	 Val. Loss: 2.170 |  Val. PPL:   8.755
Learning Rate: 0.00030685385371332307
Epoch: 03 | Time: 0m 11s
	Train Loss: 2.177 | Train PPL:   8.819
	 Val. Loss: 1.869 |  Val. PPL:   6.481
Learning Rate: 0.0004402807805699846
Epoch: 04 | Time: 0m 11s
	Train Loss: 1.922 | Train PPL:   6.836
	 Val. Loss: 1.719 |  Val. PPL:   5.579
Learning Rate: 0.0005737077074266461
Epoch: 05 | Time: 0m 11s
	Train Loss: 1.748 | Train PPL:   5.743
	 Val. Loss: 1.589 |  Val. PPL:   4.901
Learning Rate: 0.0007071346342833077
Epoch: 06 | Time: 0m 11s
	Train Loss: 1.634 | Train PPL:   5.124
	 Val. Loss: 1.541 |  Val. PPL:   4.670
Learning Rate: 0.0008405615611399692
Epoch: 07 | Time: 0m 12s
	Train Loss: 1.527 | Train PPL:   4.605
	 Val. Loss: 1.448 |  Val. PPL:   4.255
Learning Rate: 0.0009739884879966308
Epoch: 08 | Time: 0m 12s
	Train Loss: 1.451 | Train PPL:   4.269
	 Val. Loss: 1.424 |  Val. PPL:   4.152
Learning Rate: 0.0009520805187590187
Epoch: 09 | Time: 0m 12s
	Train Loss: 1.330 | Train PPL:   3.782
	 Val. Loss: 1.273 |  Val. PPL:   3.572
Learning Rate: 0.0008925569473304473
Epoch: 10 | Time: 0m 11s
	Train Loss: 1.210 | Train PPL:   3.354
	 Val. Loss: 1.229 |  Val. PPL:   3.417
Learning Rate: 0.0008330333759018759
Epoch: 11 | Time: 0m 11s
	Train Loss: 1.107 | Train PPL:   3.024
	 Val. Loss: 1.162 |  Val. PPL:   3.195
Learning Rate: 0.0007735098044733045
Epoch: 12 | Time: 0m 11s
	Train Loss: 1.035 | Train PPL:   2.815
	 Val. Loss: 1.108 |  Val. PPL:   3.028
Learning Rate: 0.000713986233044733
Epoch: 13 | Time: 0m 11s
	Train Loss: 0.951 | Train PPL:   2.589
	 Val. Loss: 1.090 |  Val. PPL:   2.975
Learning Rate: 0.0006544626616161617
Epoch: 14 | Time: 0m 11s
	Train Loss: 0.888 | Train PPL:   2.429
	 Val. Loss: 1.031 |  Val. PPL:   2.804
Learning Rate: 0.0005949390901875902
Epoch: 15 | Time: 0m 11s
	Train Loss: 0.821 | Train PPL:   2.273
	 Val. Loss: 0.986 |  Val. PPL:   2.681
Learning Rate: 0.0005354155187590188
Epoch: 16 | Time: 0m 11s
	Train Loss: 0.770 | Train PPL:   2.161
	 Val. Loss: 0.955 |  Val. PPL:   2.599
Learning Rate: 0.00047589194733044736
Epoch: 17 | Time: 0m 11s
	Train Loss: 0.717 | Train PPL:   2.048
	 Val. Loss: 0.920 |  Val. PPL:   2.509
Learning Rate: 0.000416368375901876
Epoch: 18 | Time: 0m 11s
	Train Loss: 0.663 | Train PPL:   1.940
	 Val. Loss: 0.900 |  Val. PPL:   2.460
Learning Rate: 0.00035684480447330453
Epoch: 19 | Time: 0m 11s
	Train Loss: 0.619 | Train PPL:   1.858
	 Val. Loss: 0.885 |  Val. PPL:   2.423
Learning Rate: 0.00029732123304473306
Epoch: 20 | Time: 0m 11s
	Train Loss: 0.580 | Train PPL:   1.786
	 Val. Loss: 0.852 |  Val. PPL:   2.345
Learning Rate: 0.0002377976616161617
Epoch: 21 | Time: 0m 11s
	Train Loss: 0.536 | Train PPL:   1.710
	 Val. Loss: 0.834 |  Val. PPL:   2.303
Learning Rate: 0.00017827409018759022
Epoch: 22 | Time: 0m 11s
	Train Loss: 0.507 | Train PPL:   1.660
	 Val. Loss: 0.822 |  Val. PPL:   2.276
Learning Rate: 0.00011875051875901875
Epoch: 23 | Time: 0m 11s
	Train Loss: 0.478 | Train PPL:   1.612
	 Val. Loss: 0.815 |  Val. PPL:   2.259
Learning Rate: 5.922694733044738e-05
Epoch: 24 | Time: 0m 13s
	Train Loss: 0.457 | Train PPL:   1.579
	 Val. Loss: 0.807 |  Val. PPL:   2.242
Best Model: capstone-model-23-0.8073566095395521.pt
Enter your question or type 'exit': addition
#### addition ####
------------------------------------------------------------
import itertools
def all_repeat(str1,inputstr):
	if(len(str1)==len(str1)):
		print("no")
	else:
		print("no")
------------------------------------------------------------
Enter your question or type 'exit': add two numbr
#### add two numbr ####
------------------------------------------------------------
def <unk>(a,b):
	return a+b
------------------------------------------------------------
Enter your question or type 'exit': substract two number
#### substract two number ####
------------------------------------------------------------
def <unk>(a,b):
	return a+b
------------------------------------------------------------
Enter your question or type 'exit': multiply two number
#### multiply two number ####
------------------------------------------------------------
def multiply(a,b,h):
	return(a*b+b*h)
------------------------------------------------------------
Enter your question or type 'exit': factorial
#### factorial ####
------------------------------------------------------------
num=int(input("enter a number:"))
factorial=1
if num<0:
	print("sorry,factorial does not exist for negative numbers")
elif num==0:
	print("the factorial of 0 is 1")
else:
	for i in range(1,num+1):
	factorial=factorial*i
	print("the factorial of",num,"is",factorial)
------------------------------------------------------------
Enter your question or type 'exit': fibonacci
#### fibonacci ####
------------------------------------------------------------
def fibonacci(n):
	if n<=1:
		return n
	else:
		return n+recur_fibo(n-1)
n=int(input("enter number:"))
if nterms===0:
	print("plese enter a positive integer")
else:
	print("fibonacci sequence:")
	print(nterms)
------------------------------------------------------------
Enter your question or type 'exit': fibonacci series
#### fibonacci series ####
------------------------------------------------------------
def fibonacci(n):
	if n<=1:
		return n
	else:
		return n+recur_fibo(n-1)
n=int(input("enter number:"))
if nterms===0:
	print("plese enter a positive integer")
else:
	print("fibonacci sequence:")
	print(nterms)
------------------------------------------------------------
Enter your question or type 'exit': random aray
#### random aray ####
------------------------------------------------------------
import random
print(random.randint(0,1,100))
------------------------------------------------------------
Enter your question or type 'exit': random string
#### random string ####
------------------------------------------------------------
import random
def <unk>(stringlength):
	return random.choice(name)
------------------------------------------------------------
Enter your question or type 'exit': exit
