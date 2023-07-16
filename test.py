import os
import json
import pandas as pd

label_path = "data/label.txt"

labels = pd.read_csv(label_path)

def cus_fun(x):
    id = x["id"]
    res = {
                "entities": [],
                "starts": [],
                "ends": []
            }
    return res

# print(labels)
labels["test"] = labels.apply(lambda x:cus_fun(x), axis=1)

print(labels)
