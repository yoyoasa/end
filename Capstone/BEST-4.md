# BEST-4 With Conala Dataset

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

# One cycle schedule with custome function
schedule = np.interp(np.arange(N_EPOCHS+1), [0, 5, 20, N_EPOCHS], [LEARNING_RATE, MAX_LR, LEARNING_RATE/5.0, LEARNING_RATE/10.0])
def lr_schedules(epoch):
    return schedule[epoch+1]


## Console Output

Device: cuda
Data Set Size 7015
Valid Data Size 701
Unique tokens in Question vocabulary: 1375
Unique tokens in Answer vocabulary: 3077
The model has 9,162,273 trainable parameters
Pretrained embedding dimension torch.Size([3077, 300])
Learning Rate: 0.00027
Epoch: 01 | Time: 0m 14s
	Train Loss: 3.467 | Train PPL:  32.037
	 Val. Loss: 2.389 |  Val. PPL:  10.907
Learning Rate: 0.00043999999999999996
Epoch: 02 | Time: 0m 13s
	Train Loss: 2.289 | Train PPL:   9.862
	 Val. Loss: 2.021 |  Val. PPL:   7.547
Learning Rate: 0.00061
Epoch: 03 | Time: 0m 13s
	Train Loss: 1.987 | Train PPL:   7.294
	 Val. Loss: 1.784 |  Val. PPL:   5.952
Learning Rate: 0.00078
Epoch: 04 | Time: 0m 13s
	Train Loss: 1.802 | Train PPL:   6.063
	 Val. Loss: 1.707 |  Val. PPL:   5.512
Learning Rate: 0.00095
Epoch: 05 | Time: 0m 13s
	Train Loss: 1.694 | Train PPL:   5.441
	 Val. Loss: 1.602 |  Val. PPL:   4.962
Learning Rate: 0.000888
Epoch: 06 | Time: 0m 13s
	Train Loss: 1.542 | Train PPL:   4.674
	 Val. Loss: 1.516 |  Val. PPL:   4.552
Learning Rate: 0.000826
Epoch: 07 | Time: 0m 12s
	Train Loss: 1.414 | Train PPL:   4.112
	 Val. Loss: 1.404 |  Val. PPL:   4.073
Learning Rate: 0.0007639999999999999
Epoch: 08 | Time: 0m 12s
	Train Loss: 1.290 | Train PPL:   3.632
	 Val. Loss: 1.358 |  Val. PPL:   3.889
Learning Rate: 0.0007019999999999999
Epoch: 09 | Time: 0m 11s
	Train Loss: 1.192 | Train PPL:   3.293
	 Val. Loss: 1.296 |  Val. PPL:   3.655
Learning Rate: 0.0006399999999999999
Epoch: 10 | Time: 0m 11s
	Train Loss: 1.108 | Train PPL:   3.028
	 Val. Loss: 1.224 |  Val. PPL:   3.402
Learning Rate: 0.000578
Epoch: 11 | Time: 0m 12s
	Train Loss: 1.035 | Train PPL:   2.816
	 Val. Loss: 1.173 |  Val. PPL:   3.232
Learning Rate: 0.000516
Epoch: 12 | Time: 0m 13s
	Train Loss: 0.967 | Train PPL:   2.629
	 Val. Loss: 1.142 |  Val. PPL:   3.134
Learning Rate: 0.000454
Epoch: 13 | Time: 0m 12s
	Train Loss: 0.903 | Train PPL:   2.467
	 Val. Loss: 1.081 |  Val. PPL:   2.949
Learning Rate: 0.000392
Epoch: 14 | Time: 0m 11s
	Train Loss: 0.837 | Train PPL:   2.310
	 Val. Loss: 1.051 |  Val. PPL:   2.862
Learning Rate: 0.00033
Epoch: 15 | Time: 0m 11s
	Train Loss: 0.782 | Train PPL:   2.185
	 Val. Loss: 1.021 |  Val. PPL:   2.776
Learning Rate: 0.000268
Epoch: 16 | Time: 0m 11s
	Train Loss: 0.736 | Train PPL:   2.087
	 Val. Loss: 0.983 |  Val. PPL:   2.673
Learning Rate: 0.0002059999999999999
Epoch: 17 | Time: 0m 11s
	Train Loss: 0.692 | Train PPL:   1.998
	 Val. Loss: 0.959 |  Val. PPL:   2.608
Learning Rate: 0.00014399999999999992
Epoch: 18 | Time: 0m 11s
	Train Loss: 0.656 | Train PPL:   1.927
	 Val. Loss: 0.936 |  Val. PPL:   2.550
Learning Rate: 8.199999999999993e-05
Epoch: 19 | Time: 0m 11s
	Train Loss: 0.618 | Train PPL:   1.854
	 Val. Loss: 0.920 |  Val. PPL:   2.509
Learning Rate: 2e-05
Epoch: 20 | Time: 0m 11s
	Train Loss: 0.597 | Train PPL:   1.816
	 Val. Loss: 0.914 |  Val. PPL:   2.494
Learning Rate: 1.9333333333333333e-05
Epoch: 21 | Time: 0m 13s
	Train Loss: 0.589 | Train PPL:   1.802
	 Val. Loss: 0.910 |  Val. PPL:   2.484
Learning Rate: 1.866666666666667e-05
Epoch: 22 | Time: 0m 13s
	Train Loss: 0.588 | Train PPL:   1.800
	 Val. Loss: 0.908 |  Val. PPL:   2.480
Learning Rate: 1.8e-05
Epoch: 23 | Time: 0m 12s
	Train Loss: 0.584 | Train PPL:   1.793
	 Val. Loss: 0.908 |  Val. PPL:   2.480
Learning Rate: 1.7333333333333336e-05
Epoch: 24 | Time: 0m 11s
	Train Loss: 0.578 | Train PPL:   1.783
	 Val. Loss: 0.907 |  Val. PPL:   2.476
Learning Rate: 1.6666666666666667e-05
Epoch: 25 | Time: 0m 12s
	Train Loss: 0.575 | Train PPL:   1.778
	 Val. Loss: 0.907 |  Val. PPL:   2.476
Learning Rate: 1.6000000000000003e-05
Epoch: 26 | Time: 0m 11s
	Train Loss: 0.576 | Train PPL:   1.779
	 Val. Loss: 0.903 |  Val. PPL:   2.468
Learning Rate: 1.5333333333333334e-05
Epoch: 27 | Time: 0m 11s
	Train Loss: 0.571 | Train PPL:   1.770
	 Val. Loss: 0.904 |  Val. PPL:   2.470
Learning Rate: 1.4666666666666668e-05
Epoch: 28 | Time: 0m 11s
	Train Loss: 0.568 | Train PPL:   1.764
	 Val. Loss: 0.903 |  Val. PPL:   2.466
Learning Rate: 1.4000000000000001e-05
Epoch: 29 | Time: 0m 11s
	Train Loss: 0.566 | Train PPL:   1.762
	 Val. Loss: 0.901 |  Val. PPL:   2.463
Learning Rate: 1.3333333333333335e-05
Epoch: 30 | Time: 0m 11s
	Train Loss: 0.566 | Train PPL:   1.761
	 Val. Loss: 0.900 |  Val. PPL:   2.460
Learning Rate: 1.2666666666666668e-05
Epoch: 31 | Time: 0m 13s
	Train Loss: 0.563 | Train PPL:   1.756
	 Val. Loss: 0.900 |  Val. PPL:   2.459
Learning Rate: 1.2e-05
Epoch: 32 | Time: 0m 13s
	Train Loss: 0.561 | Train PPL:   1.752
	 Val. Loss: 0.900 |  Val. PPL:   2.460
Learning Rate: 1.1333333333333334e-05
Epoch: 33 | Time: 0m 12s
	Train Loss: 0.561 | Train PPL:   1.752
	 Val. Loss: 0.898 |  Val. PPL:   2.454
Learning Rate: 1.0666666666666667e-05
Epoch: 34 | Time: 0m 11s
	Train Loss: 0.559 | Train PPL:   1.748
	 Val. Loss: 0.898 |  Val. PPL:   2.455
Learning Rate: 1e-05
Epoch: 35 | Time: 0m 11s
	Train Loss: 0.556 | Train PPL:   1.743
	 Val. Loss: 0.897 |  Val. PPL:   2.453
Best Model: capstone-model-34-0.8971251818266782.pt
Enter your question or type 'exit': add two numbers
#### add two numbers ####
------------------------------------------------------------
num1=1.5
num2=6.3
sum=num1+num2
print(f'sum:{sum}')
------------------------------------------------------------
Enter your question or type 'exit': multiplication of two numbers
#### multiplication of two numbers ####
------------------------------------------------------------
def <unk>(x,y):
	if x>y:
		smaller=x
	else:
		smaller=x
	for i in range(1,smaller+1):
		if((((((x%i==0))))and(y%i==0)):
			hcf=i
	return hcf
------------------------------------------------------------
Enter your question or type 'exit': +9658
#### +9658 ####
------------------------------------------------------------
def towerofhanoi(n):
	if n==1:
		return n
	else:
		return n+recur_fibo(n-1)+recur_fibo(n-2)
------------------------------------------------------------
Enter your question or type 'exit': tower of hennoi
#### tower of hennoi ####
------------------------------------------------------------
def towerofhanoi(n):
	if n==1:
		return n
	else:
		return(n-1)+recur_fibo(n-2)
------------------------------------------------------------
Enter your question or type 'exit': factorial
#### factorial ####
------------------------------------------------------------
def factorial(n):
	if n<=1:
		return n
	else:
		return n*factorial(n-1)
------------------------------------------------------------
Enter your question or type 'exit': exit
