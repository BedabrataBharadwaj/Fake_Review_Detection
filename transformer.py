import pandas as pd
import torch
from transformers import BertTokenizer as bt
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification as bc
from torch.utils.data import Dataset
from torch.optim import AdamW

df = pd.read_csv("review_labels.csv")
print(df.head(-10))
labels = torch.tensor(df["label"].values)

tokenizer = bt.from_pretrained("bert-base-uncased")
tokens = tokenizer(
    df["review_text"].tolist(),
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)

class TextDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels 
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.tokens.items()}
        item["labels"] = self.labels[idx]
        return item

dataset = TextDataset(tokens, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=False)
model = bc.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(3):
    for batch in loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("Epoch:", epoch + 1, "Loss:", loss.item())

model.eval()
embeddings_list = []
with torch.no_grad():
    for batch in loader:
        outputs = model.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        cls = outputs.last_hidden_state[:, 0, :]
        embeddings_list.append(cls)

embeddings = torch.cat(embeddings_list)
emb_df = pd.DataFrame(embeddings.numpy())
emb_df.to_csv("bert_embeddings.csv", index=False)
