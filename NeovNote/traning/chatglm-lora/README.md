
## ChatGLM-Tuning

- code: `https://github.com/mymusise/ChatGLM-Tuning.git`
- commit id : `997393046a49510e6cda36962f9a399297959311`



### 初始化环境

```
docker run -dt --name stanford_alpaca_v3 --restart=always --gpus all --network=host \
-v /data/cache:/root/.cache \
-v `pwd`:/workspace \
-w /workspace \
stanford_alpaca:v3 \
/bin/bash
```


### 创建 dataset（tokenize 化）

```
python ChatGLM-Tuning/tokenize_dataset_rows.py \
\--jsonl_path  data/alpaca_data_cleaned.jsonl  \
\--save_path data/alpaca_tokenize \
\--max_seq_length 200 \
\--skip_overlength True

llm-action/traning/chatglm-lora/data/alpaca_tokenize
```


### 训练
```bash
python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
```

![Alt text](image.png)

这里很奇怪用到了 WB



## Troubleshoot

### datasets.builder.DatasetGenerationError: An error occurred while generating the dataset
```
export HF_DATASETS_OFFLINE=True
python ChatGLM-Tuning/tokenize_dataset_rows.py \--jsonl_path  data/alpaca_data_cleaned.jsonl  \--save_path data/alpaca_tokenize \--max_seq_length 200 \--skip_overlength True
```


### assert vocab_file is not None
```
export https_proxy=http://192.168.100.254:7890
export HUGGING_FACE_HUB_TOKEN=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
python tokenize_dataset_rows.py \--jsonl_path  data/alpaca_data_cleaned.jsonl  \--save_path data/alpaca_tokenize \--max_seq_length 200 \--skip_overlength True
```


### NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.

Try doing:

`pip install -U datasets`

    This error stems from a breaking change in fsspec. It has been fixed in the latest datasets release (2.14.6). Updating the installation with pip install -U datasets should fix the issue.

git link : https://github.com/huggingface/datasets/issues/6352

if you are using fsspec then do:

`pip install fsspec==2023.9.2`

    There is a problem with fsspec==2023.10.0

git link : https://github.com/huggingface/datasets/issues/6330

```
pip install -U datasets
python tokenize_dataset_rows.py \--jsonl_path  data/alpaca_data_cleaned.jsonl  \--save_path data/alpaca_tokenize \--max_seq_length 200 \--skip_overlength True
```


### FileNotFoundError: Directory data is neither a `Dataset` directory nor a `DatasetDict` directory.
需要使用之前转换的输出目录
```
root@i-zigfc13j:/workspace/ChatGLM-Tuning# ls data/
.keep                      alpaca/                    alpaca_data.json           alpaca_data.jsonl          alpaca_data_cleaned.jsonl  alpaca_tokenize/
root@i-zigfc13j:/workspace/ChatGLM-Tuning# ls data/alpaca_tokenize/
data-00000-of-00001.arrow  dataset_info.json  state.json
root@i-zigfc13j:/workspace/ChatGLM-Tuning# python finetune.py     --dataset_path data/alpaca_tokenize     --lora_rank 8     --per_device_train_batch_size 6     --gradient_accumulation_steps 1     --max_steps 52000     --save_steps 1000     --save_total_limit 2     --learning_rate 1e-4     --fp16     --remove_unused_columns false     --logging_steps 50     --output_dir output
```

