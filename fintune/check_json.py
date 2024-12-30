import json
with open('output2.json', 'r') as f:

        data = json.load(f)
        print(len(data))
