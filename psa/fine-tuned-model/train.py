# Eidos777/sysprompt_judge_v0.001
#
# Training based distilgpt2
 
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("teilomillet/system_prompt")
train_dataset = dataset["train"]
eval_dataset = dataset["train"]

def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",          
    num_train_epochs=3,              
    per_device_train_batch_size=4,   
    save_steps=10_000,               
    save_total_limit=2,              
    logging_dir='./logs',            
    logging_steps=500,               
    evaluation_strategy="steps",     
    eval_steps=1000                  
)

trainer = Trainer(
    model=model,                     
    args=training_args,              
    train_dataset=train_dataset,     
    eval_dataset=eval_dataset        
)


trainer.train()

trainer.save_model("./results/final_model")

tokenizer.save_pretrained("./results/final_model")

