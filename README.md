# MiniBERT 中文情感分类（支持 MoE）

## 项目简介

本项目实现了一个 **轻量化 BERT（MiniBERT）中文情感分类模型**，并在 **Feed-Forward 层引入 Mixture of Experts (MoE)** 机制，比较了有无 MoE 在模型性能上的差异。
模型基于 **PyTorch** 从零实现，支持：

* **BERT 基础结构**（Embedding + Multi-Head Attention + FFN）
* **RMSNorm** 归一化
* **KV Cache** 机制
* **MoE FFN** 全参数专家网络（可调专家数与 Top-k）
* **中文情感分类任务**（二分类）
* **TensorBoard 日志可视化**
* **Gradio Web Demo** 推理界面

---

## 项目结构

```
MiniBERT/
│
├── config.py                  # 模型配置类 MiniBertConfig
├── model/
│   ├── attention.py           # 多头注意力（支持 KV Cache）
│   ├── block.py               # Transformer Block（可选 MoE FFN）
│   ├── embedding.py           # Token / Position / Segment Embedding
│   ├── moeffn.py              # 普通 FFN 与 MoE FFN 实现
│   ├── models.py              # MiniBERT 主体
│   ├── norm.py                # RMSNorm 层
│
├── train/
│   ├── train.py               # 训练与验证流程（支持 TensorBoard）
│
├── gradio.py                  # 推理 Demo
├── dataset/                   # 存放 parquet 格式的训练/测试数据
├── runs/                      # TensorBoard 日志文件
├── checkpoints/               # 训练过程保存的最佳模型
└── README.md
```

---

## 环境依赖

```bash
pip install torch torchvision torchaudio
pip install transformers==4.40.0
pip install pandas scikit-learn tqdm
pip install tensorboard
pip install gradio
```

---

## 数据准备

将数据集放入 `dataset/` 目录，格式为 **Parquet**，包含两列：

* `text`：中文句子
* `label`：情感标签（0=负面，1=正面）

例如：

```csv
text,label
"这部电影太精彩了",1
"剧情拖沓，无聊透顶",0
```

---

## 训练与验证

### 1. 普通 MiniBERT（无 MoE）

```python
from config import MiniBertConfig
config = MiniBertConfig(use_moe=False)
```

### 2. MiniBERT + MoE

```python
from config import MiniBertConfig
config = MiniBertConfig(use_moe=True, moe_num_experts=4, moe_top_k=1)
```

### 3. 运行训练

```bash
python train/train.py
```

训练过程会将：

* 每步训练损失 (`Train/Loss`)
* 每轮平均训练损失 (`Train/Avg_Loss`)
* 验证集准确率 (`Val/Accuracy`)
* 验证集 F1 (`Val/F1`)
  记录到 **TensorBoard**：

```bash
tensorboard --logdir=runs
```

---

## 推理 Demo

训练完成后，可运行 Gradio Web 界面：

```bash
python gradio.py
```

在浏览器中输入中文句子，模型会预测情感类别（正面/负面）。

---

## 实验结果对比

### Without MoE

* **验证集准确率**：约 82.8%
* **验证集 F1 值**：约 82.7%
* **收敛速度**：较慢
* **表现**：准确率与 F1 提升幅度有限

### With MoE

* **验证集准确率**：约 84.6%
* **验证集 F1 值**：约 84.5%
* **收敛速度**：更快
* **表现**：在相同训练轮数下，MoE 提升了约 1.8% 的准确率和 F1

---

## 可视化结果

下图为 **有/无 MoE** 两种情况下的训练与验证过程对比（取自 TensorBoard 导出）：
<h2 align="center">Without MoE</h2>

<table align="center">
  <tr>
    <td align="center"><strong>Train_Avg_Loss</strong></td>
    <td align="center"><strong>Train_Loss</strong></td>
  </tr>
  <tr>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/dff72e17-c928-4e7e-b585-6774e7a72c77" /></td>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/6e6fbaf3-1c74-4727-8a9c-bdf67a12a0a2" /></td>
  </tr>
  <tr>
    <td align="center"><strong>Val_Accuracy</strong></td>
    <td align="center"><strong>Val_F1</strong></td>
  </tr>
  <tr>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/ae0ca1f8-7c8a-449b-b862-30884977a7f6" /></td>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/72c2cd90-9856-4069-97e6-3c3fd3b73ec0" /></td>
  </tr>
</table>

---

<h2 align="center">With MoE</h2>

<table align="center">
  <tr>
    <td align="center"><strong>Train_Avg_Loss</strong></td>
    <td align="center"><strong>Train_Loss</strong></td>
  </tr>
  <tr>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/5154cf4c-5c8d-42b7-897c-38e274a83f20" /></td>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/61e24e03-16eb-4ffe-90fa-fab87d0d0554" /></td>
  </tr>
  <tr>
    <td align="center"><strong>Val_Accuracy</strong></td>
    <td align="center"><strong>Val_F1</strong></td>
  </tr>
  <tr>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/d5e74e42-cfb2-4554-ba28-016c2bd672d5" /></td>
    <td><img width="330" height="200" src="https://github.com/user-attachments/assets/e8eb064c-9780-42b7-a970-11dbc80b5399" /></td>
  </tr>
</table>

