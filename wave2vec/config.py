import os

class Path_Config():
    dataset_path = "/wav2vec2/s-kr/fine-tune/dataset"

    kss_path = os.path.join(dataset_path, "kss")
    zeroth_path = os.path.join(dataset_path, "zeroth")
    train_zeroth_path = os.path.join(zeroth_path, "train_data")
    test_zeroth_path = os.path.join(zeroth_path, "test_data")
    kspon_path = os.path.join(dataset_path, "KsponSpeech")
    output_dir = os.path.join(dataset_path, "results")

class Param_Config():
    max_sec=10.0
    min_sec=2.0
    time_limit=500*60*60
    train_rate=0.95
    pj_name='wav2vec2-south-ft'
    
    group_by_length=True
    batch_size=4
    evaluation_strategy="steps"
    num_train_epochs=100
    fp16=True
    gradient_checkpointing=True
    eval_steps=200
    learning_rate=1e-4
    log_on_each_node=True
    weight_decay=0.005
    warmup_steps=1000
    eval_accumulation_steps=1
    fsdp="full_shard"
    report_to="wandb"
    save_total_limit=200
