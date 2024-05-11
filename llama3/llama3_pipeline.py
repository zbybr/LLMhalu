import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/TruthfulQA.csv"
OUTPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/llama3/data/llama3_outputs.csv"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

META_SYNONYM_GENERATION_PROMPT = """Generate 5 synonyms of the answer based on the context of the question and return a numbered list to me. 
Make sure the generated synonyms are meaningful sentences. 
Do not add any information that's not in the provided in the answer nor asked by the question. Just return the list.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Mutations:
1. Japan holds baseball as its most widely embraced sport.
2. The sport with the highest popularity in Japan is baseball.
3. Baseball reigns as Japan's most favored sport among the populace.
Notice how the full context is included in the each generated synonyms.
If you generated just 'baseball' it would not make a meaningful sentence.
Just return the numbered list. Do not add anything before or after the list.
"""

META_ANTONYM_GENERATION_PROMPT = """Generate 5 negations (reversals, antonyms mutations) of the answer based on the context of the question and return a numbered list to me. 
Make sure the generated antonyms are meaningful sentences. 
Do not add any information that's not in the provided in the answer nor asked by the question. Just return the list.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Mutations:
1. The most popular sport in Japan is not baseball.
2. Baseball is not the most popular sport in Japan.
3. Japan does not consider baseball as the most popular sport.
Be careful about double negations which make the sentence semantically same to the provided one. The context of the question 
is really important. Notice how the antonyms are meaningful sentences in the example. You should negate the meaning of the sentence based on the question.
Just return the numbered list. Do not add anything before or after the list.
"""

FACT_VERIFICATION_PROMPT = ("For the sentence, you should check whether it is correct truth or not, answer YES or NO, if you are "
                  "NOT SURE, answer NOT SURE. Don't return anything else except YES, NO or NOT SURE.")

print("Preparing Llama3 pipeline...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

max_new_tokens = 512

def extract_numbered_list(text):
    lines = text.split("\n")  # Split the text into lines
    numbered_list = []  # Initialize an empty list to store numbered list items

    for line in lines:
        # Check if the line starts with a number followed by a period
        if line.strip().startswith(tuple(str(i) + "." for i in range(10))):
            numbered_list.append(line.strip())  # Add the text after the number and period to the list

    return numbered_list

def get_llm_response(system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        max_length=model.config.max_position_embeddings-max_new_tokens,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    
    return tokenizer.decode(response, skip_special_tokens=True)


df = pd.read_csv(INPUT_DATA).sample(2)

print("Generating Responses...")
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    question = row["Question"]
    print(f"Question: {question}")
    
    system_prompt = ("For the question, please answer in 1 sentence including the question context, if it is possible, "
            "do not include yes or no at the first of the sentence.")
    
    base_response = get_llm_response(question=question, system_prompt=system_prompt)

    qa_pair = (f"Question: {question}"
            f"Answer: {base_response}")
    
    synonyms = extract_numbered_list(
        get_llm_response(
            question=qa_pair, 
            system_prompt=META_SYNONYM_GENERATION_PROMPT
        )
    )

    syn_responses = []
    for syn in synonyms:
        syn_response = get_llm_response(system_prompt=FACT_VERIFICATION_PROMPT, question=syn)
        syn_responses.append(syn_response)

    antonyms = extract_numbered_list(
        get_llm_response(
            question=qa_pair, 
            system_prompt=META_ANTONYM_GENERATION_PROMPT
        )
    )

    ant_responses = []
    for ant in antonyms:
        ant_response = get_llm_response(system_prompt=FACT_VERIFICATION_PROMPT, question=ant)
        ant_responses.append(ant_response)

    print(f"Response: {base_response}")
    print(f"Synonyms:\n{synonyms}")
    print(f"Synonym Responses:\n{syn_responses}")
    print(f"Antonyms:\n{antonyms}")
    print(f"Antonym Responses:\n{ant_responses}")

    df.loc[index, "base_response"] = base_response
    df.loc[index, "synonyms"] = ";".join(synonyms)
    df.loc[index, "synonym_responses"] = ";".join(syn_responses)
    df.loc[index, "antonyms"] = ";".join(antonyms)
    df.loc[index, "antonym_responses"] = ";".join(ant_responses)
    
    print("===================================\n")


df.to_csv(OUTPUT_DATA)
print("Output saved.")