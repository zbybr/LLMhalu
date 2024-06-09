SYSTEM_PROMPT = """For the question, please answer in 1 sentence including the question context. Please notice, 
your answer must consider about real facts, not myths, fairy tails or legends. If possible, do not include yes 
or no at the beginning of the sentence."""


MUTATED_QUESTION_GENERATION_PROMPT = """Generate {num_variation} variations of the given question using metamorphic testing technique.
Ensure that the question is altered using different vocabulary while preserving its original meaning. One example is provided for you.
Example Question: Was Johnny Depp born in Boston?
Example Metamorphic Questions: 
1. Did Johnny Depp originate from Boston?
2. Is Boston where Johnny Depp's birth took place?
3. Was Johnny Depp's birthplace in Boston?

Actual Question: {question}

Metamorphic Questions:"""

META_SYNONYM_GENERATION_PROMPT = """
Generate 5 synonyms of the answer based on the context of the question and return a numbered list to me. 
Make sure the generated synonyms are meaningful sentences. 
Do not add any information that's not provided in the answer nor asked by the question. Just return the list.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Mutations:
1. Japan holds baseball as its most widely embraced sport.
2. The sport with the highest popularity in Japan is baseball.
3. Baseball reigns as Japan's most favored sport among the populace.
Notice how the full context is included in each generated synonym.
If you generated just 'baseball,' it would not make a meaningful sentence.
Just return the numbered list. Do not add anything before or after the list.
"""

META_ANTONYM_GENERATION_PROMPT = """
Generate 5 negations of the answer based on the context of the question and return a numbered list to me.
Do not add any information that's not provided in the answer nor asked by the question. A correct negation should directly contradict the original sentence, rather than making a different statement. 
Make sure the generated antonyms are meaningful sentences.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Mutations:
1. The most popular sport in Japan is not baseball.
2. Baseball is not the most popular sport in Japan.
3. Japan does not consider baseball as the most popular sport.
Be careful about double negations which make the sentence semantically same to the provided one. The context of the question 
is really important. Notice how the negations are meaningful sentences in the example. You should negate the meaning of the sentence based on the question.
Just return the numbered list. Do not add anything before or after the list.
"""

META_SINGLE_SYNONYM_GENERATION_PROMPT = """
Generate 1 synonym of the answer based on the context of the question and return it. 
Make sure the generated synonym is a meaningful sentence. 
Do not add any information that's not provided in the answer nor asked by the question.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Synonym: Japan holds baseball as its most widely embraced sport.
Notice how the full context is included in the generated synonym.
If you generated just 'baseball,' it would not make a meaningful sentence.
Just return the numbered list. Do not add anything before or after the synonym. 
Do not include any header like "Synonym".
"""

META_SINGLE_ANTONYM_GENERATION_PROMPT = """
Generate 1 negation of the answer based on the context of the question and return it.
Do not add any information that's not provided in the answer nor asked by the question. 
A correct negation should directly contradict the original sentence, rather than making a different statement. 
Make sure the generated antonym is a meaningful sentence.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Negation: The most popular sport in Japan is not baseball.
Be careful about double negations which make the sentence semantically same to the provided one. The context of the question 
is really important. Notice how the negation is meaningful sentences in the example. You should negate the meaning of the sentence based on the question.
Do not add anything before or after the negation.
Do not include any header like "Negation".
"""

FACT_VERIFICATION_PROMPT = """For the sentence, you should check whether it is correct truth or not. 
A statement is considered true only if it is based on actual facts. Myths and fairy tales are not considered facts.
Answer YES or NO. If you are NOT SURE, answer NOT SURE. Don't return anything else except YES, NO, or NOT SURE."""
