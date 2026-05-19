from transformers import PegasusTokenizer, PegasusForConditionalGeneration

path = "artifacts/model_trainer/finedtuned-pegasus-samsum-model"

print("Loading tokenizer...")
tokenizer = PegasusTokenizer.from_pretrained(path)

print("Loading model...")
model = PegasusForConditionalGeneration.from_pretrained(
    path,
    low_cpu_mem_usage=True
)

print("MODEL LOADED SUCCESSFULLY")

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

path = r"artifacts/model_trainer/pegasus-samsum-model"

print("Loading tokenizer...")
tokenizer = PegasusTokenizer.from_pretrained(path)

print("Loading model...")
model = PegasusForConditionalGeneration.from_pretrained(
    path,
    low_cpu_mem_usage=True
)

print("MODEL LOADED SUCCESSFULLY")

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

path = r"artifacts/model_trainer/pegasus-samsum-model"

tokenizer = PegasusTokenizer.from_pretrained(path)

model = PegasusForConditionalGeneration.from_pretrained(
    path,
    low_cpu_mem_usage=True
)

text = """
A: Hey where are you?
B: I'm at the office.
A: Can you send the report today?
B: Yes, I will send it by evening.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True
)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=40,
    min_length=10,
    num_beams=4
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\nSUMMARY:")
print(summary)