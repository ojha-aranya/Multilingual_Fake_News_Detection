import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import zipfile
import os

# file_list = []
# fake_file_path_list = [
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_assamese.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_burmese.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_gujrati.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_hindi.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_malayalam.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_marathi.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_odia.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_sinhala.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\fact_crescendo_tamil.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\factly_telugu.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\vishwas_punjabi.xlsx",
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Fake_Dataset\vishwas_urdu.xlsx"
# ]
#
# real_file_path_list = [
#     r"C:\Users\KIIT0001\Desktop\Mini_Project_Folder\Real_Dataset\Authentic-48K.xlsx"
# ]
#
# for file_path in fake_file_path_list:
#     df = pd.read_excel(file_path, engine="openpyxl")
#     df["text"] = df["title"] + df["content"]
#     df["label"] = 1
#     for index, text in df["text"].items():
#         text = str(text)
#         text = re.sub(r'<.*?>', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         df.loc[index, "text"] = text
#     file_list.append(df[["text", "label"]])
#
# fake_news_data = pd.concat(file_list, ignore_index=True)   # This is the data frame of the fake news articles
# # print(type(fake_news_data))
# # print(fake_news_data.shape)
# # print(fake_news_data.tail())
#
# df = pd.read_excel(real_file_path_list[0], engine="openpyxl")
# df = df[["headline", "content"]]
# df = df.dropna(subset=["headline", "content"], how="all")
# df["headline"] = df["headline"].fillna("").astype(str)
# df["content"] = df["content"].fillna("").astype(str)
# df["text"] = df["headline"] + " " + df["content"]
#
# df["label"] = 0
# df = df[["text", "label"]]
# for index, text in df["text"].items():
#     text = str(text)
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     df.loc[index, "text"] = text
# df = df.sample(n=fake_news_data.shape[0])
# # print(df)
#
# # This df_data is the total dataset we will be using for training
# df_data = pd.concat([fake_news_data,df], axis=0)
# df_data = df_data.sample(frac=1, random_state=42).reset_index(drop=True)
# df_data = pd.read_csv(r"/path/to/dataset/dataset_file.csv")
archive_path = r"/mnt/storage/deepak/priyasmita/Qwen3-tts/ojha/archive.zip"
extract_dir  = r"/mnt/storage/deepak/priyasmita/Qwen3-tts/ojha"
csv_path     = os.path.join(extract_dir, "dataset_file.csv")
if not os.path.exists(csv_path):
    with zipfile.ZipFile(archive_path, 'r') as z:
        z.extractall(extract_dir)
df_data = pd.read_csv(csv_path)

# df_data = df_data.sample(n=100)
# for index, text in df_data["text"].items():
#     text = str(text)
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     df_data.iloc[index,"text"] = text
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

# df_data = df_data.sample(n=100)


# Now we will generate embeddings
embeddings_cache = os.path.join(extract_dir, "embeddings_cache.pt")
if os.path.exists(embeddings_cache):
    print("\nLoading cached embeddings\n")
    embeddings_tensor = torch.load(embeddings_cache, weights_only=True, map_location='cpu')
else:
    print("\nGenerating Embeddings\n")
    embed_model = SentenceTransformer("sentence-transformers/LaBSE")
    embeddings_tensor = embed_model.encode(
        df_data["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    torch.save(embeddings_tensor, embeddings_cache)

print("Embedding Done")

# from transformers import AutoTokenizer, AutoModel

#
# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
# print("Loading model...")
# model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
# c=1
# embeddings_list = []
#
# print("Model loaded successfully!")
#
# print("Starting encoding...")
#
# for text in df_data["text"]:
#   inputs = tokenizer(text, return_tensors="pt",
#                      padding=True,
#                      truncation=True)
#   with torch.no_grad():
#     outputs = model(**inputs)
#   embeddings = outputs.last_hidden_state.mean(dim=1)
#   embeddings_list.append(embeddings)
#   print("Embedding ",c," generated")
#   c = c + 1

# print("\nConverting the list of embedding vectors into a tensor of embedding vectors\n")
# embeddings_tensor = torch.tensor(embeddings_list)  # Embedding Tensor

print("\nConverting the labels into a tensor of labels")
label_list = df_data["label"].tolist()
label_tensor = torch.tensor(label_list)   # Label Tensor

print("\nGenerating Cosine Similarity Matrix\n")
embeddings_np = embeddings_tensor.detach().cpu().numpy()
similarity_matrix = cosine_similarity(embeddings_np)
print(similarity_matrix)

threshold = 0.90  # raised from 0.75 to reduce edge count and avoid GPU OOM
rows,columns = np.where(np.triu(similarity_matrix, k=1) > threshold)
edges_list = list(zip(rows, columns))
edges_tensor = torch.tensor(edges_list)
edges_tensor = edges_tensor.t().contiguous()   #  Edges Tensor

print("\nTrain Test Split\n")

indices = range(embeddings_tensor.shape[0])
train_indx, test_indx = train_test_split(
    indices,
    test_size = 0.2,
    random_state = 42,
    stratify=df_data["label"].values
)

train_mask = torch.zeros(embeddings_tensor.shape[0],
                         dtype = torch.bool)
test_mask = torch.zeros(embeddings_tensor.shape[0],
                        dtype = torch.bool)

train_mask[train_indx] = True
test_mask[test_indx] = True

# Defining the model

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

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
model = FakeNewsGAT().to(device)
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
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(embeddings_tensor, edges_tensor)
        pred = out.argmax(dim=1)  # Get the highest probability class
        correct = (pred[test_mask] == label_tensor[test_mask]).sum()
        acc = int(correct) / int(test_mask.sum())
    return acc

print("\nStarting the Training\n")
embeddings_tensor = embeddings_tensor.clone().detach().float().to(device)
edges_tensor      = edges_tensor.to(device)
label_tensor      = label_tensor.to(device)
train_mask        = train_mask.to(device)
test_mask         = test_mask.to(device)
for epoch in range(1, 151):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Accuracy: {acc*100:.2f}%")