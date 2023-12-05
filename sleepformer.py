# %load_ext autoreload
# %autoreload 2

import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np
import pandas as pd
from tqdm import tqdm
from local_attention import LocalAttention
from sklearn.metrics import precision_score, recall_score, f1_score


def convert_timestamps_to_input(timestamps):
    # Convert the timestamps to a pandas datetime object
    timestamps = pd.to_datetime(timestamps, utc = True)

    # Extract relevant features
    features = {
        'year': timestamps.dt.year,
        'month': timestamps.dt.month,
        'day': timestamps.dt.day,
        'hour': timestamps.dt.hour,
        'minute': timestamps.dt.minute,
        'second': timestamps.dt.second
    }

    # Convert features to a PyTorch tensor
    input_tensor = np.array([features[feature].values for feature in features]).T.astype(float)

    return input_tensor


class SleepDataset(Dataset):
    def __init__(self, parquet_file, sequence_length, is_test=False):
        self.data = pd.read_parquet(parquet_file)
        self.is_test = is_test
        self.sequence_length = sequence_length
        self.series_ids = self.data['series_id'].unique()
        self.data_chunks = self.preprocess_data()

    def preprocess_data(self):
        data_chunks = []
        for series_id in self.series_ids:
            series_data = self.data[self.data['series_id'] == series_id]
            data = series_data[['anglez', 'enmo']].values.astype(np.float32)
            timestamp = convert_timestamps_to_input(series_data['timestamp'])
            step = series_data['step'].values.astype(np.float32)

            # Divide the time series into equal-sized chunks
            for i in range(0, len(data), self.sequence_length):
                chunk_data = data[i:i + self.sequence_length, :]
                chunk_timestamp = timestamp[i:i + self.sequence_length]
                chunk_step = step[i:i + self.sequence_length]

                if len(chunk_data) == self.sequence_length:  # Only include if the chunk is of sufficient length
                    if not self.is_test:
                        label = series_data['label'].values[i:i + self.sequence_length].astype(float)
                        data_chunks.append({'series_id': series_id, 'step': chunk_step, 'timestamp': chunk_timestamp, 'data': chunk_data, 'label': label})
                    else:
                        data_chunks.append({'series_id': series_id, 'step': chunk_step, 'timestamp': chunk_timestamp, 'data': chunk_data})

        return data_chunks

    def __len__(self):
        return len(self.data_chunks)

    def __getitem__(self, idx):
        return self.data_chunks[idx]


# +
class TimestepEmbedding(nn.Module):
    def __init__(self, input_size = 6, embedding_size = 128):
        super(TimestepEmbedding, self).__init__()
        self.embedding_layer = nn.Linear(input_size, embedding_size)

    def forward(self, timestep):
        # Assuming timestep is a tensor of shape (batch_size, input_size)
        embedded_timestep = self.embedding_layer(timestep)
        return embedded_timestep

class AccelerometerEmbedding(nn.Module):
    def __init__(self, input_size = 2, embedding_size = 128):
        super(AccelerometerEmbedding, self).__init__()
        self.embedding_layer = nn.Linear(input_size, embedding_size)

    def forward(self, timestep):
        embedded_accelerometer = self.embedding_layer(timestep)
        return embedded_accelerometer

class SlidingAttention(nn.Module):
    def __init__(self, embed_size, num_heads, window_size):
        super(SlidingAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size

        assert embed_size % num_heads == 0, "Embedding size must be divisible by the number of heads"
        self.head_dim = embed_size // num_heads
        self.attention = LocalAttention(dim = self.head_dim, window_size = window_size)

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask = None):
        # Split the embedding into self.num_heads different pieces
        values = values.reshape(values.shape[0], values.shape[1], self.num_heads, self.head_dim)
        keys = keys.reshape(keys.shape[0], keys.shape[1], self.num_heads, self.head_dim)
        queries = query.reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        values = self.values(values).transpose(1, 2)
        keys = self.keys(keys).transpose(1, 2)
        queries = self.queries(queries).transpose(1, 2)
        
        out = self.attention(queries, keys, values)
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.num_heads * self.head_dim)

        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, window_size, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.self_attention = SlidingAttention(input_size, num_heads, window_size)
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # Self-Attention
        attention_output = self.self_attention(x, x, x)
        x = x + self.dropout(attention_output)
        x = self.layer_norm1(x)

        # Feedforward
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x

class TransformerSlidingAttentionCRFModel(nn.Module):
    def __init__(self, timestep_input_size, accel_input_size, timestep_embedding_size, accel_embedding_size, num_heads, num_transformer_blocks, window_size, num_classes):
        super(TransformerSlidingAttentionCRFModel, self).__init__()
        self.name = 'TransformerSlidingCRFAttentionv1'
        self.num_classes = num_classes

        self.timestep_embedding = TimestepEmbedding(timestep_input_size, timestep_embedding_size)
        self.accel_embedding = AccelerometerEmbedding(accel_input_size, accel_embedding_size)
        
        # Concatenate both embeddings
        hidden_size = timestep_embedding_size + accel_embedding_size

        # Stack multiple TransformerBlocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, hidden_size, num_heads, window_size) 
            for _ in range(num_transformer_blocks)
        ])

        # Classification layer
        self.classification_layer = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes)

    def forward(self, timestep, accel_data, tags = None, mask = None):
        timestep_embedding = self.timestep_embedding(timestep)
        accel_embedding = self.accel_embedding(accel_data)
        
        combined_embedding = torch.cat([timestep_embedding, accel_embedding], dim=-1)

        # Apply multiple TransformerBlocks
        for transformer_block in self.transformer_blocks:
            combined_embedding = transformer_block(combined_embedding, mask)

        logits = self.classification_layer(combined_embedding)
        if tags is not None:
            output = -self.crf(logits, tags)
        else :
            output = self.crf.decode(logits)
        return output


# -

def train_model(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Wrap the DataLoader with tqdm to display a progress bar
    for batch in tqdm(data_loader, leave=False):
        # Extract data from the batch
        timestamp_data = batch['timestamp'].type(torch.FloatTensor).to(device)
        accelerometer_data = batch['data'].type(torch.FloatTensor).to(device)
        labels = batch['label'].type(torch.LongTensor).to(device)

        # Clear the previous gradients
        optimizer.zero_grad()

        # Forward pass
        loss = model(timestamp_data, accelerometer_data, tags = labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate accuracy and print during training
    average_loss = total_loss / len(data_loader)
    return average_loss


def test_model(model, test_loader, criterion):
    model.eval()
    loss = 0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            timestamp_data = batch['timestamp'].type(torch.FloatTensor).to(device)
            accelerometer_data = batch['data'].type(torch.FloatTensor).to(device)
            labels = batch['label'].type(torch.LongTensor).to(device)

            logits = model(timestamp_data, accelerometer_data)
            logits = torch.tensor(np.array(logits).T).contiguous().to(device)

            predicted = logits.view(-1)
            labels_flat = labels.view(-1)

            correct_predictions += torch.sum(predicted == labels_flat).item()
            total_samples += labels.numel()

            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())
    
    # Calculate validation metrics
    accuracy = correct_predictions / total_samples

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1


# Example usage for training DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8  # Adjust based on your needs
sequence_length = 250  # Adjust based on your needs

print('Loading Dataset')
dataset = SleepDataset(parquet_file='dataset/combined_0.parquet', sequence_length=sequence_length, is_test=False)
# dataset = SleepDataset(parquet_file='dataset/train_toy.parquet', sequence_length=sequence_length, is_test=False)
print('Dataset Loaded')

# Split the dataset into training and validation sets (80:20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,  # Adjust based on your system capabilities
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,  # Adjust based on your system capabilities
)

# Example usage
timestep_input_size = 6
accel_input_size = 2
timestep_embedding_size = 128
accel_embedding_size = 128
num_heads = 4
num_transformer_blocks = 4
window_size = 50
num_classes = 2
learning_rate = 0.0001
num_epochs = 10

# Initialize the model
model = TransformerSlidingAttentionCRFModel(timestep_input_size, accel_input_size, timestep_embedding_size, accel_embedding_size, num_heads, num_transformer_blocks, window_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# +
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log = open('logs/' + model.name + f'{timestamp}' + '.log', 'w')
# -

# Train the model
print('Starting Model Training')
best_f1 = 0.0
for epoch in range(num_epochs):
    loss = train_model(model, train_loader, criterion, optimizer)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    log.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}\n')
    accuracy, precision, recall, f1 = test_model(model, val_loader, criterion)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    log.write(f'Validation Accuracy: {accuracy * 100:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n')
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'models/' + model.name + '.pth')
        print('Model Saved')

log.close()


