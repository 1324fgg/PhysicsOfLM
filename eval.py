#完整代码
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import random
import argparse

# custumer code
import cfg

def main():
    # 1. 定义CFG规则
    rules3b = cfg.rules3b()

    # 2. 生成样本的函数
    def generate_sample(root):
        if root not in rules3b:
            return [root]  # 基本情况
        productions = rules3b[root]
        production = random.choice(list(productions.values()))
        result = []
        for prod in production:
            for token in prod:
                result.extend(generate_sample(token))  # 递归
        return result

    # 3. 定义数据集类
    class CFGDataset(Dataset):
        def __init__(self, num_samples=10000):
            self.samples = [generate_sample(22) for _ in range(num_samples)]
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            tokens = self.tokenizer.encode(' '.join(map(str, sample)), return_tensors='pt', max_length=512, truncation=True)
            return tokens.squeeze(0)  # 返回一维张量

    # 4. 定义模型类
    class CustomGPTModel(GPT2LMHeadModel):
        def forward(self, input_ids, attention_mask=None):
            outputs = super().forward(input_ids, attention_mask=attention_mask)
            return outputs

    # 5. 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=96,
        num_train_epochs=1,
        max_steps=100000,
        save_steps=10000,
        logging_dir='./logs',
    )

    # 6. 创建数据集和数据加载器
    train_dataset = CFGDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=96)

    # 7. 初始化模型
    model = CustomGPTModel.from_pretrained("gpt2")

    # 8. 训练模型
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    # 9. 定义评估函数
    def evaluate(model, dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs in dataloader:
                outputs = model(inputs)
                predicted = torch.argmax(outputs.logits, dim=1)  # 假设输出是logits
                # 这里假设我们有真实标签，labels需要通过某种方式定义
                # correct_predictions += (predicted == labels).sum().item()
                # total_predictions += labels.size(0)

        # accuracy = correct_predictions / total_predictions
        # return accuracy

    # 10. 进行评估（目前没有真实标签，示例代码注释掉）
    # accuracy = evaluate(model, train_dataloader)
    # print(f'Accuracy: {accuracy:.4f}')

def test():
    # testing getting cfg rules for a dataset/hardcoded
    rules = cfg.rules_ptb()

    print(rules)
    pass

if __name__ == "__main__":
    test()