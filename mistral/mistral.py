import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Mistral:
    def __init__(self, model_id="mistralai/Mistral-7B-v0.1", max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def invoke(self, system_prompt, question, temperature=0.1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            max_length=self.model.config.max_position_embeddings - self.max_new_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        response = outputs[0][input_ids.shape[-1] :]

        return self.tokenizer.decode(response, skip_special_tokens=True)
