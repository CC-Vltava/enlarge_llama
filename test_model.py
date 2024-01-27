from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModel

# 选择适当的模型和tokenizer
model_name = "/data1/cchuan/data/weight/xlmr/"  # 或者其他适用于你任务的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


inputs = tokenizer(["Hello, how are you?", "I'm doing well, thank you."], return_tensors="pt",  max_length=512, padding='max_length')
output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
print(output)
