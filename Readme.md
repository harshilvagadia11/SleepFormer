# SleepFormer: A Transformer-based Sleep Classifier

## About this Project
Refer to `report.pdf` for details on this project.
![poster](https://github.com/HV007/SleepFormer/blob/main/poster.png)

## Dataset
Download the sleep labelled accelerometer timeseries data from this <a link=https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states> link </a>. Run `Data_convert.ipynb` to convert the events into sleep states. The converted data will be used as the input to the models.

## Training the Model
SleepFormer, its variants and all other baselines are implemented as standalone notebooks. Run `<model_name>.py` to train the corresponding model. The per-epoch evaluation of the training is saved at `logs/<model_name>.log`. The best model is saved at `models/<model_name>.pth`. The `events` variable at the end stores the prediction events on the test set.

## Visualisation
To visualise the output of the model, use `sleepformer_visualise.py`.
