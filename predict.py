import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random


def predict_events(model, test_loader):
    model.eval()

    # Make predictions on the test dataset
    predictions = []
    series_ids = []
    steps = []  
    with torch.no_grad():
        for batch in tqdm(test_loader):
            test_input = batch['data']
            print(test_input)

            # Forward pass
            logits = model(test_input)

            # Convert logits to predictions
            _, predicted = torch.max(logits, 2)
            predictions.extend(predicted.cpu().numpy()[0])
            series_ids.extend(np.full(len(predicted.cpu().numpy()[0]), batch['series_id'][0]))
            steps.extend(batch['step'].cpu().numpy()[0])
            
    pred_dict = pred_to_dict(series_ids, steps, predictions)
    pred_dict = remove_outliers(pred_dict)
    pred_dict = get_local_best(pred_dict)

    prediction_rows = []
    curr_state = None
    curr_series = None
    for series_id, series_dict in tqdm(pred_dict.items()):
        curr_state = series_dict["preds"][0]
        for pred, step in zip(series_dict["preds"][1:],  series_dict["steps"][1:]):
            
            if pred != curr_state:
                curr_state = pred
                event = 'wakeup' if curr_state == 1 else 'onset'
                prediction_rows.append({
                    'row_id': len(prediction_rows),
                    'series_id': curr_series,
                    'step': int(step),
                    'event': event,
                    'score': 1.0
                })
    
    return pd.DataFrame(prediction_rows)


# +
#converts a list of parallel series_ids and predictions to a dictionary with series_ids as keys and prediction sequences as values
def pred_to_dict(series_ids, steps, predictions):
    curr_series = None
    pred_dict = {}
    curr_series = series_ids[0]
    pred_dict[curr_series] = {"preds": [], "steps": []}
    
    for series_id, step, pred in tqdm(zip(series_ids, steps, predictions)):
        if curr_series != series_id:
            curr_series = series_id
            pred_dict[curr_series] = {"preds": [], "steps": []}
       
        pred_dict[series_id]["preds"].append(pred)
        pred_dict[series_id]["steps"].append(step)
    
    return pred_dict
            
#given a dictionary of preds, drops any state boundry that does not have threshold% of one label type on the left and threshold% of the
#other label type on the right
def remove_outliers(pred_dict, window_size=1000, threshold=.8):
    new_dict = {}
    
    for series_id, series_dict in tqdm(pred_dict.items()):
        new_seq = []
        pred_seq = series_dict["preds"]
        
        for i in range(len(pred_seq)):
            if i < window_size:
                left_window = pred_seq[:i]
            else:
                left_window = pred_seq[i - window_size:i]
            
            if len(pred_seq) - i < window_size:
                right_window = pred_seq[i:]
            else:
                right_window = pred_seq[i:i+window_size]
          
            #check if pred in the middle of the window is on a state boundry
            if i > 0 and i < len(pred_seq) - window_size - 1 and left_window[-1] != right_window[0]:
                #get the proportion of labels in right window that are the same as the edge pred
                right_counts = np.bincount(right_window, minlength=2)
                r_similar = right_counts[abs(right_window[0] - 1)] / len(right_window)

                #get the proportion of labels in the left window that are different from the edge pred
                left_counts = np.bincount(left_window, minlength = 2)
                l_similar = left_counts[abs(right_window[0] - 1)] / len(left_window)

                #if the predicted label occurs frequently in both windows, then either l_similar or r_similar will be low and this will pass
                if l_similar >= threshold and r_similar >= threshold:
                    #flips state label
                    print(i, " flipped")
                    new_seq.append(abs(right_window[0] - 1))
                else:  
                    new_seq.append(right_window[0])
                                
            else:
                new_seq.append(right_window[0])
        new_dict[series_id] = {}
        new_dict[series_id]["preds"] = new_seq
        new_dict[series_id]["steps"] = series_dict["steps"]
        
    return new_dict

#Note: run remove outliers firsts so that edges are clustered around true edges
#given a dictionary of predictions, find and choose the best edge in a cluster of edges
#returns a new dictionary with one state boundry per window_size cluster
def get_local_best(pred_dict, window_size=1000):
    new_dict = {}
    
    for series_id, series_dict in tqdm(pred_dict.items()):
        pred_seq = series_dict["preds"]
        
        new_seq = [pred_seq[0]]
        i = 1
        while i < len(pred_seq) - 1:
            #if a boundry is found
            if pred_seq[i] != pred_seq[i-1]:
                #all previous labels were smooth, so find all boundries to the right inside of the window
                window = pred_seq[i: i + window_size]
                print(window)
                  
                counts = np.bincount(window, minlength = 2)
                most_common = 1 if counts[1] > counts[0] else 0
                
                indicies = [j for j in range(1, len(window)) if window[j] != most_common]
        
                #sample an index from the distribution of boundries in the cluster window
                print(indicies)
                mean = np.mean(indicies)
                std = np.std(indicies)
                print(std)
                best = int(random.gauss(mean, std))
                
                #add the previous state up to the sampled label, and the new state after the sampled label
                new_seq.extend([abs(most_common - 1)] * best)
                new_seq.extend([most_common] * (window_size - best))
                i += window_size
                
            #if label is not an edge, add to the new pred sequence
            else:
                new_seq.append(pred_seq[i])
                i += 1
                
        new_dict[series_id] = {}
        new_seq.append(pred_dict[series_id]["preds"][-1])
        new_dict[series_id]["preds"] = new_seq
        new_dict[series_id]["steps"] = series_dict["steps"]
        
    return new_dict
        
            


# +
# steps = [i for i in range(1000)]
# preds = [0 if i < 500 else 1 for i in range(1000)]
# series_ids = [1] * 1000



# preds[488] = 1
# preds[450] = 1
# preds[493] = 1
# preds[495] = 1
# preds[496] = 1
# preds[510] = 0
# preds[503] = 0
# preds[502] = 0
# preds[501] = 0
# preds[515] = 0
# preds[520] = 0
# preds[10] = 1
# preds[882] = 0
# preds[999] = 0

# pred_dict = pred_to_dict(series_ids, steps, preds)
# print(pred_dict[1]["preds"])
# pred_dict = remove_outliers(pred_dict, 20, .5)
# print(pred_dict[1]["preds"])
# print(get_local_best(pred_dict, 100)[1]["preds"])



# +
# torch.multiprocessing.set_sharing_strategy('file_system')
# saved_model_path = 'models/TransformerSlidingCRFAttentionv1.pth'  # Update with your actual saved model path

# timestep_input_size = 6
# accel_input_size = 2
# timestep_embedding_size = 128
# accel_embedding_size = 128
# num_heads = 8
# num_transformer_blocks = 2
# window_size = 50
# num_classes = 2
# learning_rate = 0.0005
# num_epochs = 10

# # Initialize the test dataset and DataLoader (replace with your actual test dataset)
# test_dataset = SleepDataset(parquet_file='dataset/test_series.parquet', sequence_length=100, is_test=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

# # Initialize the model
# input_size = 2  # Assuming 2 features in the input (anglez and enmo)
# hidden_size = 64
# num_layers = 2
# num_classes = 2  # Number of classes: asleep and awake
# dropout = 0.1

# model = BiLSTMModel(input_size, hidden_size, num_layers, num_classes, dropout)

# # Load the saved model state dictionary
# model.load_state_dict(torch.load(saved_model_path))

# # Load the model and make predictions
# pred_df = predict(model, test_loader)
                           
# print(pred_df)
# +
# pred_df.to_csv('dataset/test_predictions.csv', index=False)
# -


