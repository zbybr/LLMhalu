import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Llama3:
    def __init__(self, model_id, max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def invoke(
        self, system_prompt, question, temperature=0.5, return_token_usage=False
    ):
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

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
        )
        response = outputs[0][input_ids.shape[-1] :]

        if return_token_usage:
            prompt_tokens = len(input_ids[0])
            completion_tokens = len(outputs[0]) - prompt_tokens

            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            return (
                self.tokenizer.decode(response, skip_special_tokens=True),
                token_usage,
            )

        return self.tokenizer.decode(response, skip_special_tokens=True)
