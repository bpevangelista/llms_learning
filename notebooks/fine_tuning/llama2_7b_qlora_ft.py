# Databricks notebook source
# MAGIC %md
# MAGIC Evangelista – LLama2 7B QLoRA 4~8b v3
# MAGIC Databricks SingleNode g5.xlarge

# COMMAND ----------

# MAGIC %pip install git+https://github.com/huggingface/peft
# MAGIC %pip install -q accelerate bitsandbytes deepspeed optimum trl
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import datasets, gc, mlflow, os, sys, time, torch
from datasets import Dataset, load_dataset
from datetime import datetime
from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor
from transformers import AutoConfig, DataCollatorForLanguageModeling, LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, PeftModel, TaskType

BASE_PATH = '/dbfs/users/bruno.evangelista/'
FILESTORE_BASE_PATH = '/dbfs/FileStore/users/bruno.evangelista/'
if not os.path.exists(BASE_PATH):
  os.makedirs(BASE_PATH)

PRETRAINED_MODEL_URL = 'meta-llama/Llama-2-7b-chat-hf'
MODEL_OUTPUT_PATH = f'{BASE_PATH}/llama2/Llama-2-7b-chat-hf-fine-tune'
HF_ACCESS_TOKEN=os.environ.get('HF_ACCESS_TOKEN')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  # Using torch, silence TF
os.environ['TF_ENABLE_ONEDNN_OPTS']='0' # Using torch, silence TF

def free_mem():
  torch.cuda.empty_cache()
  gc.collect()

def print_trainable_parameters(model):
  trainable_params = 0; all_param = 0
  for _, param in model.named_parameters(): 
    all_param += param.numel()
    trainable_params += param.numel() if param.requires_grad else 0
  print(f"trainable params: {trainable_params:_} || all params: {all_param:,}")

def load_tokenizer():
  tokenizer = LlamaTokenizer.from_pretrained(
    PRETRAINED_MODEL_URL,
    token=HF_ACCESS_TOKEN,
  )
  tokenizer.pad_token = tokenizer.eos_token
  return tokenizer

def load_model(deepspeed_config=None):
  config = AutoConfig.from_pretrained(
    PRETRAINED_MODEL_URL,
    token=HF_ACCESS_TOKEN,
  )
  print('Inference Parameters:', config.to_dict())

  model = LlamaForCausalLM.from_pretrained(
    PRETRAINED_MODEL_URL,
    token=HF_ACCESS_TOKEN,
    torch_dtype=torch.float16,  # use bfloat16 if supported
    #load_in_4bit=True,
    load_in_8bit=True,          # on g4.xlarge use 4bit
    #low_cpu_mem_usage=True,
    use_flash_attention_2=True,
  )

  lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules = ['q_proj', 'v_pro'],
    task_type=TaskType.CAUSAL_LM,
  )
  
  print_trainable_parameters(model)
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  # Update deepspeed_config for model if needed here

  free_mem()
  return model

# COMMAND ----------

SYS_PROMPT = 'TODO PREENCHER EM PORTUGUES-BR'

def format_llama2(prompt: str, response: str = None, sys_prompt: str = SYS_PROMPT, tokenizer = None) -> (str, list, list):
  sys_text = f'<<SYS>>\n{sys_prompt}\n<</SYS>>\n\n' if sys_prompt is not None else ''
  prompt_text = f'[INST]{prompt}[/INST]\n\n'
  reply_text = f'{response}\n' if response is not None else ''
  
  full_prompt = f'<s>{sys_text}{prompt_text}{reply_text}</s>'
  if tokenizer is None:
    return full_prompt, None, None
  else:
    token_result = tokenizer(full_prompt, truncation=True, max_length=2048, return_tensors='pt')
    return full_prompt, token_result['input_ids'], token_result['attention_mask']

def load_training_dataset(tokenizer) -> (Dataset, Dataset):
  datasets.utils.logging.disable_progress_bar()
  data_files = {'train' : f'{FILESTORE_BASE_PATH}/training.csv'}
  dataset = load_dataset('csv', data_files=data_files)['train']

  new_dataset = {'full_prompt': [], 'input_ids': [], 'attention_mask': []}
  for item in dataset:
    full_prompt, input_ids, attention_mask = format_llama2(
      item['instruction'], response=item['response'], tokenizer=tokenizer)
    new_dataset['input_ids'].append(input_ids[0])
    new_dataset['attention_mask'].append(attention_mask[0])
    new_dataset['full_prompt'].append(full_prompt)
  dataset = dataset.from_dict(new_dataset)
  
  # debug
  print(dataset['full_prompt'][0:3])
  print(dataset['input_ids'][0:3])

  split_dataset = dataset.train_test_split(train_size = 1000, test_size=1000)
  train_tokenized_dataset = split_dataset['train']
  eval_tokenized_dataset = split_dataset['test']  
  return train_tokenized_dataset, eval_tokenized_dataset

train_dataset, eval_dataset = load_training_dataset(load_tokenizer())

# COMMAND ----------

PER_DEVICE_BATCH_SIZE=1   # Need more VRAM to increase

def fine_tune_llama2(deepspeed_config=None):
  tokenizer = load_tokenizer()
  model = load_model(deepspeed_config)

  training_args = TrainingArguments(
    #deepspeed=deepspeed_config,
    output_dir=MODEL_OUTPUT_PATH,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,

    num_train_epochs=10,
    max_steps=-1,
    gradient_accumulation_steps=1,

    learning_rate=1.5e-4,
    weight_decay=0.95,
    fp16=True,
    bf16=False,

    #gradient_checkpointing=True,     # Not compatible with LoRA?
    #prediction_loss_only=True,
    #do_eval=True,
    #evaluation_strategy='epoch',

    log_level='debug',
    logging_strategy='steps',
    logging_steps=50,
  )

  trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
      tokenizer=tokenizer,
      mlm=False, # no masking for CausalLM
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    max_seq_length=2048,
    packing=False, # True uses more VRAM
    dataset_text_field='full_prompt',
  )

  free_mem()
  trainer.train()
 
  # save model and adapter
  trainer.save_model(output_dir=MODEL_OUTPUT_PATH)
  tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
  
  # save baked model
  #model.merge_and_unload()
  #model.save_pretrained('/dbfs/users/bruno.evangelista/llama2/peft_baked')

  # one sanity test
  free_mem()
  _, input_ids, attention_mask = format_llama2('Qual a origem do universo?',
    tokenizer=tokenizer)
  
  predict_params = {
    'do_sample': True,
    'top_k': 30,
    'top_p': 0.90,
    'temperature': 0.8,
    'pad_token_id': tokenizer.eos_token_id,
    'eos_token_id': tokenizer.eos_token_id,

    'input_ids': input_ids.to('torch_extensions:0'),
    'attention_mask': attention_mask.to('torch_extensions:0'),
  }
  output_texts = tokenizer.batch_decode(model.generate(**predict_params), skip_special_tokens=True)
  print(output_texts[0])


# COMMAND ----------

!free -h
print(torch.cuda.memory_summary())

# COMMAND ----------

deepspeed_config = {
  "fp16": {
    "enabled": True
  },
  "bf16": {
    "enabled": False
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": True,
    "contiguous_gradients": True,
    "sub_group_size": 5e7,
    "reduce_bucket_size": "auto",
    "reduce_scatter": True,
    "stage3_max_live_parameters" : 1e9,
    "stage3_max_reuse_distance" : 1e9,
    "stage3_prefetch_bucket_size" : 5e8,
    "stage3_param_persistence_threshold" : 1e6,
    "stage3_gather_16bit_weights_on_model_save": True,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    }
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": False
}

# COMMAND ----------

# SingleNode 1xGPU
with mlflow.start_run() as run:
  fine_tune_llama2(deepspeed_config)

# MultiNode (1xDriver + 2xWorkers), 8xGPUs (stage 3 ZeRO)
#DeepspeedTorchDistributor(numGpus=4, nnodes=2, localMode=False, deepspeedConfig=deepspeed_config).run()

# COMMAND ----------

# MAGIC %md Loading & Testing

# COMMAND ----------

peft_model = None; model = None; free_mem()

tokenizer = LlamaTokenizer.from_pretrained(
  f'{BASE_PATH}/llama2/Llama-2-7b-chat-hf-fine-tune',
  token=HF_ACCESS_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
  f'{BASE_PATH}/llama2/Llama-2-7b-chat-hf-fine-tune',
  device_map='torch_extensions:0',
  torch_dtype=torch.float16,
  token=HF_ACCESS_TOKEN,
  use_flash_attention_2=True,
)

# COMMAND ----------

peft_model = PeftModel.from_pretrained(model, f'{BASE_PATH}/llama2/Llama-2-7b-chat-hf-fine-tune')
_, input_ids, attention_mask = format_llama2('Qual a origem do universo?',
  tokenizer=tokenizer)

predict_params = {
  'do_sample': True,
  'top_k': 30,
  'top_p': 0.90,
  'temperature': 0.8,
  'pad_token_id': tokenizer.eos_token_id,
  'eos_token_id': tokenizer.eos_token_id,

  'input_ids': input_ids.to('torch_extensions:0'),
  'attention_mask': attention_mask.to('torch_extensions:0'),
}

final_model = peft_model if peft_model is not None else model
output_texts = tokenizer.batch_decode(final_model.generate(**predict_params), skip_special_tokens=True)
print(output_texts[0])
