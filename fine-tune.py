from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

path = os.getcwd()

# Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare dataset using the 🤗 Datasets library
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

train_path = path + '/train_model.txt'
raw_datasets = load_dataset('text', data_files={'train': train_path})
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir=path + '/results',          # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,                    # number of training epochs
    per_device_train_batch_size=4,         # batch size per device during training
    warmup_steps=500,                      # number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
trainer.save_model(path + '/trained_model')

