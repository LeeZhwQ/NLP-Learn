from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request


#参数
args = Namespace(
    seed = 1234,
    data_file = "titanic.csv",
    train_size = 0.75,
    test_size = 0.25,
    num_epochs = 100,
)

url = "https://raw.githubusercontent.com/LisonEvf/practicalAI-cn/master/data/titanic.csv"
response = urllib.request.urlopen(url)
html = response.read()
with open(args.data_file,'wb') as f:
    f.write(html)