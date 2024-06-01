from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import os
from datasets import load_dataset

# Loads DailyDialog dataset
dataset = load_dataset("daily_dialog")

# Loads pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Defines special tokens
tokenizer.add_special_tokens({
    'bos_token': '',
    'eos_token': '',
    'pad_token': ''
})
model.resize_token_embeddings(len(tokenizer))

# Gets conversations from dataset
conversations = dataset['train']['dialog']

# Selects a subset of conversations (1/20th of the data)
subset_size = len(conversations) // 20
subset_conversations = conversations[:subset_size]

# Tokenizes the subset of conversational data with padding
max_length = 128  # Example value, adjust as needed based on your data
input_ids = []
attention_masks = []

for conv in subset_conversations:
    # Tokenizes input and response texts separately
    input_text, response_text = conv[0], conv[1]

    # Tokenizes input text
    input_tokenized = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_length)
    input_ids.append(input_tokenized['input_ids'])
    attention_masks.append(input_tokenized['attention_mask'])

    # Tokenizes response text
    response_tokenized = tokenizer(response_text, padding='max_length', truncation=True, max_length=max_length)
    input_ids.append(response_tokenized['input_ids'])
    attention_masks.append(response_tokenized['attention_mask'])

# Converts lists to tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

# Prepares the dataset
class ConversationalDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_masks[idx]}

dataset = ConversationalDataset(input_ids, attention_masks)

# Defines training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    report_to=None,
)

# Data collator for padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Defines the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tunes the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine-tuned-gpt2')
tokenizer.save_pretrained('./fine-tuned-gpt2')