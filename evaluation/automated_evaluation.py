import sys

sys.path.append("/home/mdafifal.mamun/research/LLMhalu/llama3")

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from llama3 import Llama3


def check_similarity_by_llm(model, base_sentence, ref_sentences):
    system_prompt = "Given two sentences, your job is to tell whether or not they refer to the same meaning." \
                    "Reply SAME if the sentences mean same thing, NOT SAME if they are different in meaning." \
                    "Do not reply anything else."
    
    for ref_sentence in ref_sentences:
        question = f"Sentence 1: {base_sentence}\nSentence 2: {ref_sentence}"
        model_response = model.invoke(system_prompt, question)
        print("LLM Response:", model_response)

        if model_response.lower().strip() == "same":
            return True

    return False

def check_similarity_by_cosine_sim(model, base_sentence, ref_sentences, threshold = 0.8):
    embeddings1 = model.encode(base_sentence, convert_to_tensor=True)
    embeddings2 = model.encode(ref_sentences, convert_to_tensor=True)

    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    print(cosine_similarity)
    
    if (cosine_similarity >= threshold).any():
        return True
    else:
        return False

plm = SentenceTransformer('bert-base-nli-mean-tokens')
llm = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")

input_data = "/home/mdafifal.mamun/research/LLMhalu/llama3/data/llama3_outputs_check_2.csv"
output_data = "/home/mdafifal.mamun/research/LLMhalu/llama3/data/llama3_outputs_check_auto.csv"
df = pd.read_csv(input_data)

for index, row in df.iterrows():
    answer = row["answer"]
    print(answer)
    correct_answers = row["correctanswer"].split(";")
    print(correct_answers)

    llm_sim = check_similarity_by_llm(llm, answer, correct_answers)
    plm_sim = check_similarity_by_cosine_sim(plm, answer, correct_answers)

    # If both language model returns True then it is high-likely that the LLM responded
    # correct answer, whereas if both of them are False, it is high-likely that the LLM
    # returned a hallucinated answer. If the language models have different responses
    # a manual check may be required.

    if llm_sim == True and plm_sim == True:
        df.loc[index, "auto_hallucination_check"] = "No"
        df.loc[index, "manual_check_required"] = "No"
    
    elif llm_sim == False and plm_sim == False:
        df.loc[index, "auto_hallucination_check"] = "Yes"
        df.loc[index, "manual_check_required"] = "No"
    
    else:
        df.loc[index, "auto_hallucination_check"] = "Not Sure"
        df.loc[index, "manual_check_required"] = "Yes"


df.to_csv(output_data)