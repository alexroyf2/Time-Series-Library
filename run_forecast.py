import os
import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# Define arguments
args = argparse.Namespace(
    # Basic config
    task_name='long_term_forecast',
    is_training=1,
    model_id='sales_quantity_forecast',
    model='TimesNet',
    
    # Data loader
    data='custom',
    root_path='./dataset/custom/',
    data_path='sales_data.csv',
    features='M',  # multivariate 
    target='Quantity_Sold',
    freq='m',  # monthly
    checkpoints='./checkpoints/',
    
    # Forecasting task
    seq_len=24,  # use 2 years of data
    label_len=12,  # label length
    pred_len=12,  # forecast 12 months ahead
    seasonal_patterns=None,
    inverse=False,
    
    # Model parameters
    enc_in=3,  # number of input features
    dec_in=3,  # number of input features
    c_out=1,   # number of output features
    d_model=64,
    n_heads=4,
    e_layers=2,
    d_layers=1,
    d_ff=128,
    factor=3,
    top_k=5,
    
    # Optimization
    num_workers=0,
    train_epochs=10,
    batch_size=16,
    patience=3,
    learning_rate=0.001,
    des='Sales_Quantity_Forecast',
    loss='MSE',
    lradj='type1',
    use_amp=False,
    
    # GPU
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    devices='0',
    
    # Other defaults
    embed='timeF',
    dropout=0.1,
    activation='gelu',
    output_attention=False,
    do_predict=True,
    moving_avg=25,
    distil=True,
    decomp_method='moving_avg',
    
    # Added for other models
    itr=1,
    use_norm=1,
    num_kernels=6,
    d_conv=4,
    expand=2,
    channel_independence=1,
    down_sampling_layers=0,
    down_sampling_window=1,
    down_sampling_method=None,
    seg_len=96
)

# Set random seed
torch.manual_seed(2021)

# Create experiment
exp = Exp_Long_Term_Forecast(args)

# Train model
print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.train(setting=args.model_id)

# Test model
print('>>>>>>>testing>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.test(setting=args.model_id)

# Make predictions
print('>>>>>>>predicting>>>>>>>>>>>>>>>>>>>>>>>>>>')
exp.predict(setting=args.model_id, load=True)

print('Finished! Check predictions in the "pred" directory within checkpoints.')