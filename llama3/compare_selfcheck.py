import pandas as pd
import spacy
import torch
from dotenv import load_dotenv
from openai import OpenAI
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

nlp = spacy.load("en_core_web_sm")

client = OpenAI()


# Constants
INPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/llama3/data/llama3_outputs_check.csv"
OUTPUT_DATA = "/home/mdafifal.mamun/research/LLMhalu/llama3/data/llama3_outputs_check_2.csv"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_prompt = SelfCheckLLMPrompt(MODEL_ID, device)

# Initializing Llama3 pipeline
print("Preparing Llama3 pipeline...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
max_new_tokens = 512

def extract_numbered_list(text):
    return [line.strip() for line in text.split("\n") if line.strip().startswith(tuple(str(i) + "." for i in range(10)))]

def get_llm_response(system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        max_length=model.config.max_position_embeddings - max_new_tokens,
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


if __name__ == "__main__":    
    df = pd.read_csv(INPUT_DATA, encoding='latin-1')

    print("Generating Responses...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing issue"):
        question = row["question"]
        answer = row["answer"]

        sentences = [sent.text.strip() for sent in nlp(answer).sents]
        print(f"Question: {question}")

        system_prompt = ("For the question, please answer in 1 sentence including the question context, if possible. "
                            "Do not include yes or no at the beginning of the sentence.")
        
        generated_samples = []
        for i in range(10):
            base_response = get_llm_response(question=question, system_prompt=system_prompt)
            generated_samples.append(base_response)

        df.loc[index, "generated_samples"] = ";".join(generated_samples)

        sent_scores_prompt = selfcheck_prompt.predict(
            sentences=sentences,
            sampled_passages=generated_samples,
            verbose=True,
        )
        print(sent_scores_prompt)
        df.loc[index, "Selfcheck Scores"] = sent_scores_prompt

        print("===================================\n")

    # Save output data
    df.to_csv(OUTPUT_DATA)
    print("Output saved.")