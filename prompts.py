SYSTEM_PROMPT = """For the question, please answer in 1 sentence including the question context. Please notice, 
your answer must consider about the reality, not the myths, fairy tails or legends. If possible, do not include yes 
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

META_SYNONYM_GENERATION_PROMPT = """Generate 5 synonyms of the answer based on the context of the question and return a numbered list to me. 
Make sure the generated synonyms are meaningful sentences. 
Do not add any information that's not in the provided in the answer nor asked by the question. Just return the list.
For example:
Question: What is the most popular sport in Japan?
Answer: Baseball is the most popular sport in Japan.
Mutations:
1. Japan holds baseball as its most widely embraced sport.
2. The sport with the highest popularity in Japan is baseball.
Notice how the full context is included in the each generated synonyms.
If you generated just 'baseball' it would not make a meaningful sentence.
Also, if there is a saying sentence in the answer, you should not modify it.
For example:
Answer: Michael Cunningham wrote "You cannot find peace by avoiding life" in "The Hours".
Mutation: "You cannot find peace by avoiding life" was wrote by Michael Cunningham.
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
Be careful about double negations which make the sentence semantically same to the provided one. The context of the question 
is really important. Notice how the antonyms are meaningful sentences in the example. 
Also, you should negate the meaning of the sentence based on the question.
Please notice, if there is a saying sentence in the answer, you should not modify it.
For example:
Answer: Michael Cunningham wrote "You cannot find peace by avoiding life" in "The Hours".
Mutation: Michael Cunningham did not write "You cannot find peace by avoiding life" in "The Hours"
Do not add any information that's not in the provided in the answer nor asked by the question.
"""

FACT_VERIFICATION_PROMPT = """For the sentence, you should check whether it is correct truth in reality or not. 
Please notice, all truth in myths, fairy tails and legends will not be considered as reality. Answer YES or NO. If you 
are NOT SURE, answer NOT SURE. Don't return anything else except YES, NO, or NOT SURE."""
