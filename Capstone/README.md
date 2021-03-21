# Capstone Project


### Data Cleaning

First and the formost process in any machine learning problem is to look into data, the dataset consista of questions and their equivalent codes

#### These were following issues with data set  

1. Some questons started with # and others started with numbers so we had to make them all start with a question like syntax like `write a program`
2. Also to identify that if a perticular statement is a question or a comment we had to write rules to clean comments up first
3. Indentation problem, some programs were having tabs some had spaces and the other had mixed characters like tab, spaces

Following are some of the rules used for cleaning up dataset
1. `Manually Preprocess Data ie remove any extra '# ' from data set`
2. `<space> def` -> `def`
3. `#write` -> `# write`
4. `\n#\s?\d+` -> `\n# `  // Convert all numbered statements to un numbered
5. `\d+\.\s?\n# write` -> `# write`
6. `# Define` -> `# write`
7. `\n#.python ? 3? ?` -> `\n# write `
8. `\n#.Write (?!a)` -> `\n# write a`
9. `\n#.Write` -> `\n# write`
10. `#.program` -> `# write a program`
11. `\n\s*\n\s*\n+` -> `\n\n`,
12. `\n?# In\[\d+\]:\s*\n?` -> `\n`
13. `\n\s*\n(?!#.write)` -> `\n\n`
14. `Remove all un necessary comments`
15. `Remove all un necessary new lines`
16. `Remove all extra spcaes and replace them with tabs`
17. `Split each program into pairs of statement and code`
18. `(#.program|#.write)`

### Tokenization

As ths is a unique dataset where there are a couple number of special characters and other keywords which needs to be together

#### Following were some of the considerations needed to take in picture while writing a tokenizer
1. names can have `_` (underscore)
2. there are characters like `\n` and `\t` they needs to be a single word

To cater to these needs we had to write out custom tokenizer extending `spacy` tokenizer

    ```
    from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
    from spacy.tokenizer import Tokenizer

    def custom_tokenizer(nlp):
        infix_re = re.compile(
            r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'\(\)\[\]\{\}\*\%\^\+\-\=\<\>\|\!(//)(\n)(\t)~]''')
        prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                        suffix_search=suffix_re.search,
                        infix_finditer=infix_re.finditer,
                        token_match=None)


    spacy_que = spacy.load('en_core_web_sm')
    spacy_ans = spacy.load('en_core_web_sm')
    spacy_ans.tokenizer = custom_tokenizer(spacy_ans)


    def tokenize_que(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_que.tokenizer(text)]


    def tokenize_ans(text):
        """
        Tokenizes Code text from a string into a list of strings
        """
        return [tok.text for tok in spacy_ans.tokenizer(text)]
    ```

### Pretrained Embeddings

It is already proven that using pretrained embedding makes network learn faster and converge easier with a less number ot epochs

#### We trained a custom glove embedding with 300 dimensions 

we trained this glove embedding for 100 epochs and reached a descent loss (9.974439516472449e-13)

![Loss Graph](assets/loss.png)


#### Using these pretrained embeddings

1. While building vocabulary for answers we added these pretrained embedding vocabulary
    ```
    Answer.build_vocab(train_data, vectors=torchtext.vocab.Vectors("./python_code_glove_embedding_300.txt"), min_freq=2)
    ```

2. After applying initial weights we updated decoder embeddings with pretrained embeddings
    ```
    glove_pretrained_embeddings = Answer.vocab.vectors
    model.decoder.tok_embedding.weight.data = glove_pretrained_embeddings.to(device)
    ```

We also experimented with embeddings size of 50, 100, 250 but they din't work well so finally ended using 300 dimension embeddings

### Hyper parameters

We experimented with a couple number of hyper parameters

1. First approach was to get a stable network which is powefull enough or has a capacity to learn souch a complex dataset. We started with a small network and then with a number of experiments er reached at a network with  following parameter

    ```
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
    ```

2. After finalizing on network size we experimented on batch size which is optimal enough for network to learn better, experiment included batch size ranging from `4` to `512` and finally ended up with `32` as optimal batch size for network

3. One of the most important hyper parameter is learning rate which took a bit of time for us to arrive a a good number, we experimented a number of sterategy but finally ended up with one cucle lr as the best suited for us. To fine tune one cycle `lr` we used grid search with manual numbers we use following lines of code to add manual one cycle lr 

    ```
    # One cycle schedule with custome function
    schedule = np.interp(np.arange(N_EPOCHS+1), [0, 5, 20, N_EPOCHS], [LEARNING_RATE, MAX_LR, LEARNING_RATE/5.0, LEARNING_RATE/10.0])
    def lr_schedules(epoch):
        return schedule[epoch+1]
    ```

    Also added following code to training loop to update learning rate based on epochs

    ```
    optimizer.param_groups[0]['lr'] = lr_schedules(epoch)
    ```

    Finally after multiple iterations we ended up with following numbers

    ```
    LEARNING_RATE: 0.0005
    MAX_LR: 0.001
    N_EPOCHS: 24
    CLIP: 1
    STEPS_PER_EPOCH: 220
    ```

    then we used these numbers in pytorch onc cycle lr sterategy with linear analying to train our network

    ```
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=STEPS_PER_EPOCH, epochs=N_EPOCHS, anneal_strategy='linear')
    ```

    Also we added following code after optimizer step to step scheduler after every batch

    ```
    optimizer.step()

    scheduler.step()
    ```

### Optimizer and Loss Function

We tried with different functions but notting worked better than the default ones, we tried label smoothing but it made not a big difference as results were not so great


### Dataset extension

To get a better and robust network we used a thirdparty dataset called `conala` dataset this is open sourse dataset, this gave us a very beautiful improvement on accuracy for out existing data.
### Training Logs

We trained our networ a couple number of times and we saved some of logs as best logs you can find some of them [here](/logs), the best one for us was the one which I want to highlight here

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


### Some of example outputs

1.  addition of two number
    ```
    def add(num1,num2):
        sum=num1+num2
        return sum
    ```

2. write a python program that would print"Hello World" ####
    ```
    unicodestring=u"hello world!"
    print(unicodestring)
    ```

3. program that would merge two dictionaries by adding the second one into the first
    ```
    a={"a":1,"b":2,"c":3}
    b={"c":4,"d":3}
    a.update(b)
    ```

4. python program that asks for user input and prints the given input
    ```
    a=input("user input")
    print(a)
    ```

5. square of a given input number
    ```
    def square(x):
        return x**2
    ```

6. sum of first n natural numbers,where n is the input
    ```
    def <unk>(n):
        if n<=0:
            return n
        else:
            return n+sum_of_nums(n-1)
    ```

7. input list sorted in descending order
    ```
    def sort_descending(list_to_be_sorted):
        return sorted(list_to_be_sorted,reverse=true)
    ```

8. write a python program that would print the first n positive integers using a for loop
    ```
    def reverse_integer(n):
        if n<=0:
            return n
        else:
            return n+recursive_sum(n-1)
    ```

9. write a python fuction to create an empty list
    ```
    def emptylist():
        return list()
    ```

10. write a python function to add elements of two lists
    ```
    def append_lists(l1:list,l2:list)->list:
        return l1.extend(l2)
    ```

11. sum of two number
    ```
    def add_two_numbers(num1,num2):
        sum=num1+num2
        return sum
    ```

12. write a python function that accepts a dictionary that has unique values and returns its inversion
    ```
    def invert_dict(input_dict):
        my_inverted_dict={value:key for key,value in input_dict.items()}
        return my_inverted_dict
    ```

13. write a python function that accepts a list of dictionaries and sorts it by a specified key
    ```
    def sort_dict_list(dict_list,sort_key):
        dict_list.sort(key=lambda item:item.get(sort_key))
    ```

14. write a funtion that accepts two lists of equal length and converts them into a dictioinary 
    ```
    def list_to_dict(list1,list2):
        return dict(zip(list1,list2))
    ```

15. write a program to print number of characters in a string
    ```
    str1="pynative"
    digitcount=0
    for i in range(0,len(str1)):
        char=str1[i]
        if(char.isdigit()):
            digitcount+=1
    print('number of digits:',digitcount)
    ```

16. write a function to check if a lower case letter exists in a given string
    ```
    def check_lower(str1):
        for char in str1:
            k=char.islower()
            if k==true:
                return true
        return false
    ```

17. write function for finding the derivative of sine angle
    ```
    import math
    def sin(angle):
        return math.cos(angle)
    ```

18. write a Python function To Calculate Volume OF Cylinder
    ```
    def cal_cylinder_volume(height,radius):
        pi=3.14
        return pi*radius*height*height
    ```

19. write program to find if given co-ordinates are inside circle
    ```
    import math
    radius=5
    print(f'area:{math.pi*radius}')
    ```

20. write function for factorial using reduce
    ```
    def factorial(n):
        if n==1:
            return n
        else:
            return n*factorial(n-1)
    ```

21. write function for datetime
    ```
    from datetime import datetime
    date_string="feb 25 2020   4:20pm"
    datetime_object=datetime.strptime(date_string,'%b%d%y%i:%m%p')
    print(datetime_object)
    ```

22. write function to return the cubes of a list of numbers
    ```
    def cube(l):
        return[i***i for i in range(l)]
    ```

23. write function to find the area of a circle
    ```
    def findarea(r):
        pi=3.142
        return pi*r*r
    ```

24. write function to add even mubers in a list
    ```
    def add_even_num(l1,l2):
        return[i*j for i in l1 if i%2==0]
    ```

25. write ReLu function
    ```
    def relu(x:list)->float:
        return x if x<0 else x
    ```

26. write function to return the squares of a list of numbers
    ```
    def <unk>(nums):
        return[i*i for i in nums]
    ```

27. write function to return the nth fibonacci number
    ```
    def fib(n):
        if n<=1:
            return n
        else:
            return n+fib(n-1)
    ```

28. write a program to Shuffle a list randomly
    ```
    from random import shuffle
    mylist=[3,6,7,8]
    shuffle(mylist)
    print(mylist)
    ```

29. write a program to Test if string starts with H
    ```
    word="hello world"
    check=word.startswith('h')
    print(f"string starts with h?:{check}")
    ```

30. write a program to Replacing a string with another string
    ```
    str1="hello!it is a good thing"
    substr1="good"
    substr2="bad"
    replaced_str=str1.replace(substr1,substr2)
    print("string after replace:",str1)
    ```

31. write a program to Join Two Sets
    ```
    a={1,2,3,4,5}
    b={4,5,6,7,8}
    print(a & b)
    ```

32. write a program to keep only the items that are present in both sets
    ```
    x={"apple","banana","cherry"}
    y={"google","microsoft","apple"}
    x.intersection_update(y)
    print(f"duplicate value in two set:{x}")
    ```

33. write Program to Find HCF 
    ```
    def compute_hcf(x,y):
        if x>y:
            smaller=y
        else:
            smaller=x
        for i in range(1,smaller+1):
            if((((x%i==0))and(y%i==0)):
                hcf=i
        return hcf
    ```
    
34. write Program to Find LCM
    ```
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
    ```