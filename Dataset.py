import pandas as pd
from sklearn.model_selection import train_test_split

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
