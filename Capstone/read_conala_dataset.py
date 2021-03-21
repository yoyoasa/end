import json

dataset = []
data = {"q": "", "a": ""}

with open('./conala-corpus/conala-test.json', 'r') as f:
    records = json.load(f)
    for record in records:
        data['q'] = record['intent']
        data['a'] = record['snippet']
        dataset.append(data)

with open('./conala-corpus/conala-train.json', 'r') as f:
    records = json.load(f)
    for record in records:
        data['q'] = record['intent']
        data['a'] = record['snippet']
        dataset.append(data)


