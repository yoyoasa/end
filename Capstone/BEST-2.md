## Best-2

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

# One Cycle Scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS, anneal_strategy='linear')

##### -------------------- Console out put -------------------
Device: cuda
Data Set Size 4135
Valid Data Size 413
Unique tokens in Question vocabulary: 1382
Unique tokens in Answer vocabulary: 3152
The model has 9,209,448 trainable parameters
Pretrained embedding dimension torch.Size([3152, 300])
Learning Rate: 4e-05
Epoch: 01 | Time: 0m 9s
	Train Loss: 5.819 | Train PPL: 336.780
	 Val. Loss: 4.271 |  Val. PPL:  71.618
Learning Rate: 0.0001734917993819824
Epoch: 02 | Time: 0m 9s
	Train Loss: 3.754 | Train PPL:  42.682
	 Val. Loss: 3.309 |  Val. PPL:  27.345
Learning Rate: 0.0003069835987639648
Epoch: 03 | Time: 0m 8s
	Train Loss: 3.106 | Train PPL:  22.328
	 Val. Loss: 2.843 |  Val. PPL:  17.168
Learning Rate: 0.00044047539814594727
Epoch: 04 | Time: 0m 8s
	Train Loss: 2.726 | Train PPL:  15.269
	 Val. Loss: 2.616 |  Val. PPL:  13.685
Learning Rate: 0.0005739671975279297
Epoch: 05 | Time: 0m 8s
	Train Loss: 2.463 | Train PPL:  11.739
	 Val. Loss: 2.409 |  Val. PPL:  11.124
Learning Rate: 0.000707458996909912
Epoch: 06 | Time: 0m 8s
	Train Loss: 2.256 | Train PPL:   9.545
	 Val. Loss: 2.311 |  Val. PPL:  10.087
Learning Rate: 0.0008409507962918945
Epoch: 07 | Time: 0m 9s
	Train Loss: 2.102 | Train PPL:   8.185
	 Val. Loss: 2.148 |  Val. PPL:   8.566
Learning Rate: 0.0009744425956738769
Epoch: 08 | Time: 0m 9s
	Train Loss: 1.971 | Train PPL:   7.175
	 Val. Loss: 2.066 |  Val. PPL:   7.890
Learning Rate: 0.0009518723943833944
Epoch: 09 | Time: 0m 8s
	Train Loss: 1.794 | Train PPL:   6.012
	 Val. Loss: 1.959 |  Val. PPL:   7.094
Learning Rate: 0.0008923488229548229
Epoch: 10 | Time: 0m 9s
	Train Loss: 1.640 | Train PPL:   5.154
	 Val. Loss: 1.867 |  Val. PPL:   6.472
Learning Rate: 0.0008328252515262516
Epoch: 11 | Time: 0m 9s
	Train Loss: 1.490 | Train PPL:   4.436
	 Val. Loss: 1.792 |  Val. PPL:   6.002
Learning Rate: 0.0007733016800976801
Epoch: 12 | Time: 0m 9s
	Train Loss: 1.364 | Train PPL:   3.912
	 Val. Loss: 1.660 |  Val. PPL:   5.262
Learning Rate: 0.0007137781086691086
Epoch: 13 | Time: 0m 8s
	Train Loss: 1.253 | Train PPL:   3.501
	 Val. Loss: 1.640 |  Val. PPL:   5.155
Learning Rate: 0.0006542545372405373
Epoch: 14 | Time: 0m 9s
	Train Loss: 1.164 | Train PPL:   3.204
	 Val. Loss: 1.570 |  Val. PPL:   4.805
Learning Rate: 0.0005947309658119658
Epoch: 15 | Time: 0m 9s
	Train Loss: 1.079 | Train PPL:   2.941
	 Val. Loss: 1.552 |  Val. PPL:   4.721
Learning Rate: 0.0005352073943833944
Epoch: 16 | Time: 0m 9s
	Train Loss: 0.997 | Train PPL:   2.710
	 Val. Loss: 1.488 |  Val. PPL:   4.430
Learning Rate: 0.00047568382295482307
Epoch: 17 | Time: 0m 9s
	Train Loss: 0.930 | Train PPL:   2.534
	 Val. Loss: 1.483 |  Val. PPL:   4.406
Learning Rate: 0.0004161602515262516
Epoch: 18 | Time: 0m 8s
	Train Loss: 0.865 | Train PPL:   2.375
	 Val. Loss: 1.454 |  Val. PPL:   4.279
Learning Rate: 0.0003566366800976801
Epoch: 19 | Time: 0m 9s
	Train Loss: 0.813 | Train PPL:   2.255
	 Val. Loss: 1.431 |  Val. PPL:   4.184
Learning Rate: 0.00029711310866910876
Epoch: 20 | Time: 0m 8s
	Train Loss: 0.758 | Train PPL:   2.134
	 Val. Loss: 1.405 |  Val. PPL:   4.077
Learning Rate: 0.00023758953724053729
Epoch: 21 | Time: 0m 9s
	Train Loss: 0.712 | Train PPL:   2.038
	 Val. Loss: 1.397 |  Val. PPL:   4.043
Learning Rate: 0.00017806596581196581
Epoch: 22 | Time: 0m 8s
	Train Loss: 0.670 | Train PPL:   1.955
	 Val. Loss: 1.406 |  Val. PPL:   4.078
Learning Rate: 0.00011854239438339445
Epoch: 23 | Time: 0m 9s
	Train Loss: 0.639 | Train PPL:   1.895
	 Val. Loss: 1.378 |  Val. PPL:   3.969
Learning Rate: 5.901882295482298e-05
Epoch: 24 | Time: 0m 9s
	Train Loss: 0.615 | Train PPL:   1.849
	 Val. Loss: 1.376 |  Val. PPL:   3.959
Enter your question or type 'exit': factorial
#### factorial ####
------------------------------------------------------------
def factorial(n):
	if n==1:
		return n
	else:
		return n*factorial(n-1)
------------------------------------------------------------
Enter your question or type 'exit': addition
#### addition ####
------------------------------------------------------------
v1=[1,2,3]
v2=[3,4,5]
s1=[0,0,0,0,0,0]
for i in range(len(v1)):
	s1[i]=v1[i]
	s1[i]+v2[i]
print(f"after vector:{s1}")
------------------------------------------------------------
Enter your question or type 'exit': addition of two numbers
#### addition of two numbers ####
------------------------------------------------------------
s1=[2,3,4,5,6,7,12]
s2=list(s1)
print("length of common length of both are:"+str(s1))
------------------------------------------------------------
Enter your question or type 'exit': add two numbers
#### add two numbers ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'sum:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': multiply two numbers
#### multiply two numbers ####
------------------------------------------------------------
def multiply(x,y):
	return x*y
------------------------------------------------------------
Enter your question or type 'exit': substract two numbers
#### substract two numbers ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'product:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': divide two numbers
#### divide two numbers ####
------------------------------------------------------------
def compute_hcf(x,y):
	if x>y:
		smaller=x
	else:
		smaller=x
	for i in range(1,smaller+1):
		if(((x%i==0)and(y%i=0)):
					hcf=i
	return hcf
------------------------------------------------------------
Enter your question or type 'exit': fibonacci
#### fibonacci ####
------------------------------------------------------------
def recur_fibo(n):
	if n<=1:
		return n
	else:
		return n+recur_fibo(n-1)
nterms=[2,3,4,5]
print(recur_fibo(n-1))
------------------------------------------------------------
Enter your question or type 'exit': exit
