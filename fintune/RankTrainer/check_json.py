import json

with open('fixed.json', 'r') as f:
    try:
        data = json.load(f)
        print(data[2]['responses'][0])
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")


