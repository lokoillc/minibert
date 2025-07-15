from torch.utils.tensorboard import SummaryWriter  # 新增导入
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from config import MiniBertConfig
from model.models import MiniBertModel
import torch.nn.functional as F
import os

# === 模型包装 ===
class MiniBertForSequenceClassification(nn.Module):
    def __init__(self, config: MiniBertConfig, num_labels: int):
        super().__init__()
        self.bert = MiniBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        cls_token = outputs["last_hidden_state"][:, 0, :]  # [CLS]
        logits = self.classifier(cls_token)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# === 自定义 Dataset ===
class ParquetTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        df = pd.read_parquet(file_path)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# === 评估函数 ===
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        preds = outputs["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MiniBertConfig()
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    train_path = "..//dataset//train-00000-of-00001.parquet"
    test_path = "..//dataset//test-00000-of-00001.parquet"

    train_dataset = ParquetTextDataset(train_path, tokenizer)
    test_dataset = ParquetTextDataset(test_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = MiniBertForSequenceClassification(config, num_labels=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    best_f1 = 0.0

    writer = SummaryWriter(log_dir="runs/miniBert_MoE_experiment")  # TensorBoard写日志对象

    global_step = 0

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # 写入每个step的loss到TensorBoard
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # 评估
        acc, f1 = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}] Val Acc: {acc:.4f} | F1: {f1:.4f}")

        # 写入验证集指标到TensorBoard
        writer.add_scalar("Val/Accuracy", acc, epoch)
        writer.add_scalar("Val/F1", f1, epoch)
        writer.add_scalar("Train/Avg_Loss", avg_loss, epoch)

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_moe_model.pt")
            print("✅ Best model saved.")

    writer.close()

if __name__ == "__main__":
    train()
