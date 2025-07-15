import torch
from transformers import BertTokenizer
from config import MiniBertConfig
from model.models import MiniBertModel
# decoder-style 模型专用
# === 1. 配置与模型加载 ===
# === 2. 使用Huggingface官方Tokenizer (与BertEmbeddings兼容) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("C:\\Users\\Administrator\\PycharmProjects"
                                          "\\minibert\\tokenizer\\bert-base-chinese")  # tokenizer路径

config = MiniBertConfig(vocab_size=tokenizer.vocab_size)
model = MiniBertModel(config).to(device)
model.eval()  # 关闭Dropout等

# === 3. 输入文本 ===
full_sentence = "我 很 喜欢 这 部 电影"
part1 = "我 很"
part2 = "喜欢 这 部 电影"

# === 4. 全量输入推理 ===
full_inputs = tokenizer(full_sentence, return_tensors='pt', padding=True, truncation=True)
full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

with torch.no_grad():
    full_outputs = model(full_inputs['input_ids'],
                         token_type_ids=full_inputs.get('token_type_ids'),
                         attention_mask=full_inputs.get('attention_mask'),
                         use_cache=False)
    full_last_hidden = full_outputs['last_hidden_state']

print("=== 全量模式输出 ===")
print(full_last_hidden[:, -1, :])  # 取最后一个token表示

# === 5. 分步 + KV Cache 推理 ===

# 第一步：输入前半句
part1_inputs = tokenizer(part1, return_tensors='pt', padding=True, truncation=True)
part1_inputs = {k: v.to(device) for k, v in part1_inputs.items()}

with torch.no_grad():
    part1_outputs = model(part1_inputs['input_ids'],
                          token_type_ids=part1_inputs.get('token_type_ids'),
                          attention_mask=part1_inputs.get('attention_mask'),
                          use_cache=True)
    past_kvs = part1_outputs['past_kvs']

# 第二步：输入后半句 + KV Cache
part2_inputs = tokenizer(part2, return_tensors='pt', padding=False, truncation=True)
part2_inputs = {k: v.to(device) for k, v in part2_inputs.items()}

with torch.no_grad():
    part2_outputs = model(part2_inputs['input_ids'],
                          token_type_ids=part2_inputs.get('token_type_ids'),
                          attention_mask=part2_inputs.get('attention_mask'),
                          use_cache=True,
                          past_kvs=past_kvs)
    part2_last_hidden = part2_outputs['last_hidden_state']

print("=== KV Cache模式输出 ===")
print(part2_last_hidden[:, -1, :])  # 取后半句最后一个token表示

# === 6. 比较差距 ===
# 注意：由于BERT不是生成式模型，拆句后位置不同，cache验证主要看能否正常跑通（位置不同本身不等价）
diff = torch.abs(full_last_hidden[:, -1, :] - part2_last_hidden[:, -1, :]).mean()
print("两种方式最后Token平均差距：", diff.item())
