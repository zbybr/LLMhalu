import pandas as pd


df = pd.read_excel("D:\\Projects\\LLMhalu\\datasets\\FreshQA.xlsx")
# df = df[1:]

df.to_csv("D:\\Projects\\LLMhalu\\datasets\\freshqa.csv")