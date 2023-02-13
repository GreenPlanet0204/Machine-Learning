# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# load the dataset
dataset = pd.read_csv('dataset/Market_Basket_Optimization.csv')
transactions = []
for i in range(0, 7500):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# train the apriori model on the dataset
rules = apriori(transactions=transactions, min_support=0.003,
                min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# display the result of the rules occurred from the apriori function
results = list(rules)

# visualize the rule, support, confidence, lift in more clear way
for item in results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + "->" + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("==========================================")
