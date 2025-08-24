from transformers import BertTokenizer, BertModel
import torch

# Đường dẫn tới thư mục chứa model đã tải từ hf-mirror
model_path = "/workspace/nhitny/mcs/MoCL-NAACL-2024/pretrained/bert-base-uncased1"

# Load tokenizer và model
tokenizer = BertTokenizer.from_pretrained(model_path, use_fast=True)
model = BertModel.from_pretrained(model_path)

# Câu test
text = "Hugging Face is awesome!"

# Tokenize và convert thành tensor
inputs = tokenizer(text, return_tensors="pt")

# Dự đoán bằng model
with torch.no_grad():
    outputs = model(**inputs)

# In kích thước output để xác nhận model hoạt động
print("✅ Model output shape:", outputs.last_hidden_state.shape)
