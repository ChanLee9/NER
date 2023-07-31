import matplotlib.pyplot as plt
import json
import numpy as np

data_path = "saved_results/BERT_SPAN/BERT_SPAN_results.json"
data = json.load(open(data_path, "r"))

Ps = data["precision"]
Rs = data["recall"]
F1s = data["f1_score"]

def plot_results(Ps, Rs, F1s, save_path):
    assert len(Ps) == len(Rs) == len(F1s), "length of Ps, Rs, F1s should be equal!"
    plt.figure(figsize=(10, 5))
    xx = np.arange(1, len(Ps)+1)
    plt.plot(xx, Ps, label='precision')
    plt.plot(xx, Rs, label='recall')
    plt.plot(xx, F1s, label='f1_score')
    plt.plot(xx, np.mean(F1s)*np.ones(len(F1s)), label='average f1_score', linestyle='--')
    plt.legend()
    plt.savefig(save_path)
    
save_path = "test.png"
plot_results(Ps, Rs, F1s, save_path)
