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
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            timestamp_data = batch['timestamp'].type(torch.FloatTensor).to(device)
            accelerometer_data = batch['data'].type(torch.FloatTensor).to(device)

            logits = model(timestamp_data, accelerometer_data)
            logits = torch.tensor(np.array(logits).T).contiguous().to(device)

            predicted = logits.view(-1)
            all_preds.extend(predicted.view(-1).cpu().numpy())

    return all_preds


# Example usage for training DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8  # Adjust based on your needs
sequence_length = 250  # Adjust based on your needs

print('Loading Dataset')
dataset = SleepDataset(parquet_file='dataset/visualise.parquet', sequence_length=sequence_length, is_test=False)
# dataset = SleepDataset(parquet_file='dataset/train_toy.parquet', sequence_length=sequence_length, is_test=False)
print('Dataset Loaded')

test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

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
torch.multiprocessing.set_sharing_strategy('file_system')
saved_model_path = 'models/TransformerSlidingCRFAttentionv1.pth'  # Update with your actual saved model path
model = TransformerSlidingAttentionCRFModel(timestep_input_size, accel_input_size, timestep_embedding_size, accel_embedding_size, num_heads, num_transformer_blocks, window_size, num_classes).to(device)
model.load_state_dict(torch.load(saved_model_path))

# +
pre_preds = test_model(model, test_loader)
pre_preds = np.array(pre_preds)

round_func = np.vectorize(lambda i: 1 if i > 0.5 else 0)
preds  = round_func(savgol_filter(pre_preds, window_length=2001, polyorder=2))
# -

filtered_df = pd.read_parquet('dataset/visualise.parquet')

event_df = pd.read_csv('dataset/train_events.csv')

# +
event_df['timestamp'] = pd.to_datetime(event_df['timestamp'], utc=True)
desired_series_id = '038441c925bb'

filtered_events = event_df[
    (event_df['series_id'] == desired_series_id) & 
    (event_df['timestamp'] >= '2018-08-17 00:00:00') & 
    (event_df['timestamp'] <= '2018-08-21 23:59:59')
]

# +
# Plotting the data

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

ax1.plot(filtered_df['timestamp'].values, filtered_df['anglez'].values, label='Anglez', marker='o', color='gold')
ax1.set_ylabel('Anglez')
ax1.legend()


ax2.scatter(filtered_df['timestamp'].values, filtered_df['enmo'].values, label='ENMO', marker='x', color='lightgreen')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('ENMO')
ax2.legend()

# Separate legends for 'onset' and 'wakeup'
onset_legend = False
wakeup_legend = False

for _, event in filtered_events.iterrows():
    linestyle = '--' if event['event'] == 'onset' else '-'
    
    ax1.axvline(x=event['timestamp'], color='red', linestyle=linestyle, linewidth=2, zorder=20)
    ax2.axvline(x=event['timestamp'], color='red', linestyle=linestyle, linewidth=2, zorder=20)

    # Setting label only if it's the first occurrence
    if event['event'] == 'onset' and not onset_legend:
        ax1.axvline(x=event['timestamp'], color='red', linestyle=linestyle, linewidth=2, label='onset', zorder=20)
        ax2.axvline(x=event['timestamp'], color='red', linestyle=linestyle, linewidth=2, label='onset', zorder=20)
        onset_legend = True
    elif event['event'] == 'wakeup' and not wakeup_legend:
        ax1.axvline(x=event['timestamp'], color='red', linestyle=linestyle, linewidth=2, label='wakeup', zorder=20)
        ax2.axvline(x=event['timestamp'], color='red', linestyle=linestyle, linewidth=2, label='wakeup', zorder=20)
        wakeup_legend = True

ax1.fill_between(filtered_df['timestamp'].values, min(filtered_df['anglez'].values), max(filtered_df['anglez'].values),
                 where=(preds == 1), color='blue', alpha=0.5, label='Prediction: Asleep', zorder=10)
ax2.fill_between(filtered_df['timestamp'].values, min(filtered_df['enmo'].values), max(filtered_df['enmo'].values),
                 where=(preds == 1), color='blue', alpha=0.5, label='Prediction: Asleep', zorder=10)

# plt.suptitle(f'Accelerometer Data for series_id: {desired_series_id}')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualise_post.png')
# -




