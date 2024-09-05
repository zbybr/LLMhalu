import torch
from transformers import AutoTokenizer, pipeline


class Mistral:
    def __init__(self, model_id, max_new_tokens=512, temperature=0.1):
        if temperature:
            self.chatbot = pipeline(
                "text-generation",
                model=model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device="cuda",
                do_sample=True,
            )
        else:
            self.chatbot = pipeline(
                "text-generation",
                model=model_id,
                max_new_tokens=max_new_tokens,
                device="cuda",
            )

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def invoke(self, system_prompt, question, return_token_usage=False):
        prompt = f"{system_prompt}\n{question}"
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.chatbot(messages)
        response_content = response[0]["generated_text"][1]["content"]

        if return_token_usage:
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            num_prompt_tokens = prompt_tokens.shape[-1]

            completion_tokens = self.tokenizer(response_content, return_tensors="pt")[
                "input_ids"
            ]
            num_completion_tokens = completion_tokens.shape[-1]

            token_usage = {
                "prompt_tokens": num_prompt_tokens,
                "completion_tokens": num_completion_tokens,
                "total_tokens": num_prompt_tokens + num_completion_tokens,
            }

            return response_content, token_usage

        return response_content
