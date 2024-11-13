import pandas as pd
from sklearn.model_selection import train_test_split
import json
from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import random


# 设置设备（如果有GPU，则使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def same_seeds(seed):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

same_seeds(0)

# 加載JSON數據文件
with open('/mnt/md0/chen-wei/zi/Dimensional-ASTE/NYCU_NLP_113A_Dataset/NYCU_NLP_113A_TrainingSet.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 將JSON數據加載到Pandas DataFrame中
df = pd.json_normalize(data)

# 將數據分割為訓練集和測試集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# 為了適配BERT模型，我們需要將Sentence、Aspect、Opinion、Category和Intensity字段提取出來，並將其格式化為模型的輸入形式。


def format_for_bert(row):
    # 組合每一行的 Sentence 和 Aspect-Opinion 信息
    sentence = row['Sentence']
    aspects = row['Aspect']
    opinions = row['Opinion']
    categories = row['Category']
    intensities = row['Intensity']

    # 將Aspect, Opinion等合併為 "aspect-opinion-category-intensity" 格式
    aspect_opinion_pairs = []
    for aspect, opinion, category, intensity in zip(aspects, opinions, categories, intensities):
        aspect_opinion_pairs.append(f"{aspect}-{opinion}-{category}-{intensity}")

    # 最終組成格式化的輸入
    formatted_input = f"{sentence} | {'; '.join(aspect_opinion_pairs)}"
    return formatted_input

# 應用格式化函數到DataFrame
train_df['formatted_input'] = train_df.apply(format_for_bert, axis=1)
test_df['formatted_input'] = test_df.apply(format_for_bert, axis=1)


# 確認標籤數量
unique_labels = set()
for _, row in df.iterrows():
    for category in row['Category']:
        unique_labels.add(category)

num_labels = len(unique_labels)
print("num_labels:", num_labels)

model_name = 'bert-base-chinese'

# 使用BERT快速標記器來自動對齊標記和標籤
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

#要將句子和標籤編碼，並確保每個Aspect和Opinion的範圍能夠正確地對應到模型輸入的編碼位置。這樣可以讓模型學習如何在句子中找到正確的Aspect和Opinion標註

def encode_data(df):
    # 用於保存編碼結果
    input_ids = []
    attention_masks = []
    labels = []

    # 遍歷數據中的每一行
    for _, row in df.iterrows():
        # 獲取句子和標註
        sentence = row['Sentence']
        aspects = row['Aspect']
        categories = row['Category']

        opinions = row['Opinion']
        intensities = row['Intensity']
        
        # 使用標記器對句子進行編碼
        encoded = tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

        # 初始化標籤，使用-100忽略不需要的標籤 表示模型不會計算這些位置的損失（通常用於填充位置）。
        label = [-100] * 128

        # 處理每個Aspect和Opinion，並將標籤與標記對齊 檢查每個標記的開始和結束位置，並將對應於Aspect和Opinion的標記設置為不同的標籤（例如，1 表示 Aspect，2 表示 Opinion）。
        for aspect, opinion, category, intensity in zip(aspects, opinions, categories, intensities):
            # 獲取Aspect和Opinion的開始和結束位置
            aspect_start, aspect_end = map(int, row['AspectFromTo'].split('#'))
            opinion_start, opinion_end = map(int, row['OpinionFromTo'].split('#'))
            
            # 確保標記化後的位置正確對應
            tokens = tokenizer(sentence, return_offsets_mapping=True)
            for idx, (offset_start, offset_end) in enumerate(tokens['offset_mapping']):
                if offset_start == aspect_start and offset_end <= aspect_end:
                    label[idx] = 1  # 用1表示Aspect
                elif offset_start == opinion_start and offset_end <= opinion_end:
                    label[idx] = 2  # 用2表示Opinion

        # 添加到最終的編碼數據列表中
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        labels.append(label)

    # 最終返回 BERT 需要的 input_ids、attention_mask 和 labels，這些可以直接用於 BERT 的訓練。
    return input_ids, attention_masks, labels

# 對訓練集和測試集進行編碼
# 对训练集和测试集进行编码
train_input_ids, train_attention_masks, train_labels = encode_data(train_df)
test_input_ids, test_attention_masks, test_labels = encode_data(test_df)

# 创建TensorDataset
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)


# 步骤 5：设置训练参数



# 设置批量大小和训练轮数
batch_size = 16
epochs = 3
learning_rate = 5e-5

# 将列表转换为张量
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.tensor(train_labels)

test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.tensor(test_labels)

# 创建TensorDataset
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# 创建DataLoader
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)


# 初始化优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

# 计算总训练步数
total_steps = len(train_dataloader) * epochs

# 创建学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 步骤 6：训练模型


# 将模型移动到设备上（GPU或CPU）
model.to(device)

# 训练循环
for epoch in range(epochs):
    print(f'正在训练第 {epoch + 1} 个epoch，共 {epochs} 个')
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        # 将批次中的数据移动到设备上
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        # 清除之前的梯度
        model.zero_grad()

        # 前向传播
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)

        # 获取损失值
        loss = outputs.loss

        # 累积损失
        total_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 更新学习率
        scheduler.step()

    # 计算平均训练损失
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'平均训练损失: {avg_train_loss}')

    # 验证模型
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    nb_eval_steps = 0

    for batch in test_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)

        # 获取损失值和预测结果
        loss = outputs.loss
        logits = outputs.logits

        eval_loss += loss.item()

        # 将预测结果和标签移动到CPU并转为numpy数组
        predictions = torch.argmax(logits, dim=2).detach().cpu().numpy()
        labels = batch_labels.to('cpu').numpy()

        # 计算每个token的准确度，忽略标签为-100的部分
        for i in range(len(labels)):
            valid_labels = labels[i] != -100
            eval_accuracy += np.sum(predictions[i][valid_labels] == labels[i][valid_labels]) / np.sum(valid_labels)
            nb_eval_steps += 1

    avg_eval_loss = eval_loss / len(test_dataloader)
    avg_eval_accuracy = eval_accuracy / nb_eval_steps
    print(f'验证损失: {avg_eval_loss}')
    print(f'验证准确度: {avg_eval_accuracy}')
