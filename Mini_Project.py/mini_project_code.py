import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, f1_score

df_data = pd.read_csv(r"C:\Users\KIIT0001\Desktop\DatasetGeneration\dataset_file.csv")

# df_data = df_data.sample(n=100) # For debugging purpose
df_data["text"] = (
    df_data["text"]
        .astype(str)
        .str.replace(r"<.*?>", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
)
print(df_data)
print(df_data.shape[0])
print(df_data.shape)

print("\nGenerating Embeddings\n")

model = SentenceTransformer("sentence-transformers/LaBSE")   # Language Agnostic BERT Sentence Embedding Model

embeddings_tensor = model.encode(
    df_data["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True
)
embeddings_tensor = embeddings_tensor.clone().detach().float()

print("Embedding Generating Done")

# Train Test Split
print("\nTrain Test Split\n")

indices = range(embeddings_tensor.shape[0])
train_indx, test_indx = train_test_split(
    indices,
    test_size = 0.2,
    random_state = 42,
    stratify=df_data["label"].values
)

train_mask = torch.zeros(embeddings_tensor.shape[0],
                         dtype=torch.bool)
test_mask = torch.zeros(embeddings_tensor.shape[0],
                        dtype=torch.bool)

train_mask[train_indx] = True
test_mask[test_indx] = True

print("\nConverting the labels into a tensor of labels")
label_list = df_data["label"].tolist()
label_tensor = torch.tensor(label_list)   # Label Tensor
print(label_tensor)

print("\nGenerating Cosine Similarity Matrix\n")
embeddings_np = embeddings_tensor.detach().cpu().numpy()
similarity_matrix = cosine_similarity(embeddings_np)

# Creating Edges
threshold = 0.75
# threshold = 0.3  # For debugging purpose
rows,columns = np.where(np.triu(similarity_matrix, k=1) > threshold)
X_set = set(train_indx)
Y_set = set(test_indx)
mask = [
    (r in X_set and c in X_set) or (r in Y_set and c in Y_set)
    for r, c in zip(rows, columns)
]
rows_filtered = rows[mask]
columns_filtered = columns[mask]
edges_list = list(zip(rows_filtered, columns_filtered))
edges_tensor = torch.tensor(edges_list)
edges_tensor = edges_tensor.t().contiguous()   # Edges Tensor

class FakeNewsGAT(torch.nn.Module) :
  def __init__(self):
    super(FakeNewsGAT, self).__init__()
    self.conv1 = GATConv(in_channels=768,
                         out_channels=128,
                         heads=4,
                         dropout=0.5)
    self.conv2 = GATConv(in_channels=128*4,
                         out_channels=64,
                         heads=1,
                         dropout=0.5)
    self.classifier = torch.nn.Linear(64,2)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = F.elu(x)
    x = F.dropout(x,
                  p=0.5,
                  training=self.training)

    x = self.conv2(x,
                   edge_index)
    x = F.elu(x)
    x = F.dropout(x,
                  p=0.5,
                  training=self.training)

    out = self.classifier(x)
    return out

model = FakeNewsGAT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()



# Defining the Training Testing

def train():
    model.train()
    optimizer.zero_grad()  # Clear old gradients
    out = model(embeddings_tensor, edges_tensor)  # Feed the whole graph
    loss = criterion(out[train_mask], label_tensor[train_mask])  # Only calculate loss on Training nodes
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights


    # Train Acc
    preds = out[train_mask].argmax(dim=1)
    correct = (preds == label_tensor[train_mask]).sum().item()
    train_acc = correct / train_mask.sum().item()
    return loss.item(), train_acc

def test():
    model.eval()
    with torch.no_grad():
        out = model(embeddings_tensor, edges_tensor)
        pred = out.argmax(dim=1)  # Get the highest probability class
        correct = (pred[test_mask] == label_tensor[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
    return acc

print("\n\n--------------Starting the Training-----------------\n")
# embeddings_tensor = embeddings_tensor.clone().detach().float()
for epoch in range(1, 151):
    loss, train_acc = train()
    if epoch % 10 == 0:
        test_acc = test()
        # print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Accuracy: {acc*100:.2f}%")
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Train Acc: {train_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}%")
# Finding Zero Class Precision & Real Class Precision
y_true = label_tensor[test_mask].detach().numpy() #true labels

out = model(embeddings_tensor, edges_tensor)
out = out[test_mask]
y_pred = out.detach().numpy()
y_pred = (y_pred > 0.5).astype(int)

# print(y_true)
# print(y_pred)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

precision_fake = TP / (TP + FP)
precision_real = TN / (TN + FN)

print("\nDisplaying the Precision for Fake and real class seperately\n")
print("Fake News Precision:", precision_fake)
print("Real News Precision:", precision_real)

print("\nClassification Report :\n")
print(classification_report(y_true, y_pred))

# Recall & F Measure
recall_fake = recall_score(y_true, y_pred, pos_label=1)
recall_real = recall_score(y_true, y_pred, pos_label=0)

f1 = f1_score(y_true, y_pred)

print("Fake Recall:", recall_fake)
print("Real Recall:", recall_real)
print("F1 Score:", f1)