from transformers import pipeline


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

    def invoke(self, system_prompt, question):
        prompt = f"{system_prompt}\n{question}"
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = self.chatbot(messages)

        return response[0]["generated_text"][1]["content"]
