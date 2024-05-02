import pandas as pd
import numpy as np


SEED = 42
SAMPLES = 100
np.random.seed(SEED)


df = pd.read_csv("TruthfulQA.csv")

df = df.sample(SAMPLES)
df.to_csv(f"TruthfulQA_{SAMPLES}-samples_{SEED}-seed.csv")
