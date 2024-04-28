import spacy
import torch
from dotenv import load_dotenv
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

load_dotenv()

nlp = spacy.load("en_core_web_sm")

passage = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Nation."
sentences = [sent.text.strip() for sent in nlp(passage).sents]

sample1 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Country."
sample2 = "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. He works at The New York Times."
sample3 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He obtained his PhD from MIT."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")

sent_scores_prompt = selfcheck_prompt.predict(
    sentences = sentences,
    sampled_passages = [sample1, sample2, sample3],
    verbose = True,
)

print(sent_scores_prompt)