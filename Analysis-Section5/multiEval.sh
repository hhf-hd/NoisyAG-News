
nohup python train_bert_noise_metrics_multieval.py \
    --label noise_instance_med \
    --epochs 20 \
    --early-epochs 3 \
    --early-evals 8 \
    --mid-epochs 4 \
    --mid-evals 4 \
    --output noise_instance_med-9-29.csv >> tt_noise_instance_med.log 2>&1 &