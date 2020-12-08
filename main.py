import pandas as pd
import SRCNN_training as sr

df = pd.read_csv('train.csv')
sr.training(df, 2, 10000)