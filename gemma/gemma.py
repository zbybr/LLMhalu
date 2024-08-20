from transformers import pipeline


class Gemma:
    def __init__(self, model_id, max_new_tokens=256, temperature=0.1):
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

    def invoke(self, system_prompt, question):
        prompt = f"{system_prompt}\n{question}"
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.chatbot(messages)

        return response[0]["generated_text"][-1]["content"].strip()



# import torch
# from transformers import pipeline

# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-2-9b-it",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",  # replace with "mps" to run on a Mac device
# )

# messages = [
#     {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
# ]

# outputs = pipe(messages, max_new_tokens=256)
# assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
# print(assistant_response)
# # Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜
