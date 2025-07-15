import torch
from transformers import BertTokenizer
from config import MiniBertConfig
from train.train import MiniBertForSequenceClassification
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载 Tokenizer & Config
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
config = MiniBertConfig(vocab_size=tokenizer.vocab_size)

# 2. 加载模型
model = MiniBertForSequenceClassification(config, num_labels=2)
model.load_state_dict(torch.load("C:\\Users\\Administrator\\PycharmProjects\\minibert\\"
                                 "train\\checkpoints\\best_moe_model.pt", map_location=device))
model.to(device)
model.eval()


# 3. 推理函数
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        predicted_class = logits.argmax(dim=-1).item()

    return "正面 😊" if predicted_class == 1 else "负面 😞"


# 4. Gradio界面

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="请输入中文句子，例如：这部电影真好看！"),
    outputs=gr.Label(num_top_classes=2),
    title="MiniBERT 中文情感分类 Demo",
    description="输入任意中文句子，模型将判断其情感（正面/负面）。"
).launch()
