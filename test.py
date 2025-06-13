from transformers import AutoTokenizer

# DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# check special tokens
print("PAD Token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)
print("CLS Token:", tokenizer.cls_token, "ID:", tokenizer.cls_token_id)
print("SEP Token:", tokenizer.sep_token, "ID:", tokenizer.sep_token_id)
print("EOS Token:", tokenizer.eos_token, "ID:", tokenizer.eos_token_id)
