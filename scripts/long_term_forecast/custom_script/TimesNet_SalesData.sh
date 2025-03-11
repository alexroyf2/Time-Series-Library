export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/custom/ \
  --data_path sales_data.csv \
  --model_id sales_quantity_forecast \
  --model $model_name \
  --data custom \
  --features M \
  --target Quantity_Sold \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --d_model 64 \
  --d_ff 128 \
  --freq m \
  --des 'Sales_Quantity_Forecast' \
  --itr 1 \
  --top_k 5 