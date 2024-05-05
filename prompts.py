MUTATED_QUESTION_GENERATION_PROMPT = """Generate {num_variation} variations of the given question using metamorphic testing technique.
Ensure that the question is altered using different vocabulary while preserving its original meaning. One example is provided for you.
Example Question: Was Johnny Depp born in Boston?
Example Metamorphic Questions: 
1. Did Johnny Depp originate from Boston?
2. Is Boston where Johnny Depp's birth took place?
3. Was Johnny Depp's birthplace in Boston?

Actual Question: {question}

Metamorphic Questions:"""