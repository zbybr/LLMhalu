import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/TruthfulQA_100-samples_42-seed.csv"
OUTPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/llama3_outputs.csv"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Preparing Llama3 pipeline...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

max_new_tokens = 512
system_prompt = ("For the question, please answer in 1 sentence including the question context, if it is possible, "
            "do not include yes or no at the first of the sentence.")

def get_llm_response(question):
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


df = pd.read_csv(INPUT_DATA)[:5]

print("Generating Responses...")

responses = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
    question = row["Question"]
    print(f"Question: {question}")
    response = get_llm_response(question=question)
    print(f"Response: {response}")
    print("===================================\n")
    response = response.replace("\n\n", "\n")
    responses.append(response)


df["LLaMA3_responses"] = responses
df.to_csv(OUTPUT_DATA)
print("Output saved.")