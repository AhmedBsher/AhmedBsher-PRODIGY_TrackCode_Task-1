import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

# Load sample data
data = {
    'query': [
        "My order hasn't arrived yet. Can you provide an update?",
        "I received a damaged product. How can I get a replacement?",
        "What are your return policies for online purchases?"
    ],
    'response': [
        "I apologize for the delay. Let me check the status of your order for you.",
        "I'm sorry to hear that. Please provide your order details, and we'll arrange a replacement.",
        "Our return policy allows returns within 30 days of purchase. Could you please provide your order number?"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split into train, validation, test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Save to CSV files
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('val_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

# Load datasets
train_df = pd.read_csv('train_dataset.csv')
val_df = pd.read_csv('val_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# Initialize the tokenizer and model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Optional: Add a special token for padding if not already in tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Tokenize the data
def tokenize_function(examples):
    # Tokenize each query and response individually
    query_encodings = tokenizer(examples['query'].tolist(), max_length=512, padding="max_length", truncation=True)
    response_encodings = tokenizer(examples['response'].tolist(), max_length=512, padding="max_length", truncation=True)
    
    # Combine query and response encodings to create input and label pairs
    input_ids = [q + r for q, r in zip(query_encodings['input_ids'], response_encodings['input_ids'])]
    attention_mask = [q + r for q, r in zip(query_encodings['attention_mask'], response_encodings['attention_mask'])]
    labels = input_ids.copy()  # Use input_ids as labels for language modeling
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

train_encodings = tokenize_function(train_df)
val_encodings = tokenize_function(val_df)
test_encodings = tokenize_function(test_df)

# Convert to PyTorch Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = TextDataset(train_encodings)
val_dataset = TextDataset(val_encodings)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
