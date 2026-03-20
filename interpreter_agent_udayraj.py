import os
import json
import numpy as np
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

INTENTS = [
    "Product Comparison", "Churn Prediction", "Product Feedback", "Network Outages",
    "Credit Limit Requests", "Competitor comparison", "Price Sensitivity",
    "Delivery Delays", "Product Returns", "Replacement vs Refund",
    "Cross-Brand Mentions", "Brand Loyalty", "Discounts & Promotions",
    "Plan Upgrades", "Upgrade Requests", "Connectivity Complaints",
    "Feature Requests", "Sales effectiveness", "Fee Complaints",
    "Loan Application", "Loyalty Program", "Policy renewal",
    "Delay Management", "Fraud Alerts", "Customer trust",
    "Refund Delays", "Service Complaints", "Technical Support",
    "Claims & refunds", "Urgency & Stress", "Feature understanding",
    "Refund Policy", "Booking Errors", "Upselling strategy",
    "Cancellation Policies"
]

NUM_LABELS = len(INTENTS)
id2label = {i: label for i, label in enumerate(INTENTS)}
label2id = {label: i for i, label in enumerate(INTENTS)}
MODEL_NAME = "microsoft/deberta-v3-base"

print("Loading dataset...")
data = {"text": [], "label": []}

with open("customer_lines_with_intent.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        intent = item.get("intent")
        if intent in label2id:
            data["text"].append(item["input_text"])
            data["label"].append(label2id[intent])

hf_dataset = Dataset.from_dict(data)
hf_dataset = hf_dataset.cast_column("label", ClassLabel(num_classes=NUM_LABELS, names=INTENTS))

print("Performing Stratified 70/15/15 Split (Train/Val/Test)")

initial_split = hf_dataset.train_test_split(
    test_size=0.3, 
    seed=42, 
    stratify_by_column="label" 
)
train_dataset = initial_split["train"]
temp_dataset = initial_split["test"]

final_split = temp_dataset.train_test_split(
    test_size=0.5, 
    seed=42, 
    stratify_by_column="label" 
)
val_dataset = final_split["train"]
test_dataset = final_split["test"]

print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

print("\nInitializing DeBERTa-v3 Tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Calculating optimal sequence length")
lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in data["text"]]
optimal_max_length = int(np.percentile(lengths, 99))
optimal_max_length = min(optimal_max_length, 512) 
print(f"Setting dynamic max_length to: {optimal_max_length} tokens.")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=optimal_max_length)

total_cores = os.cpu_count() or 1
optimal_workers = min(16, max(1, total_cores - 2))

print(f"Tokenizing data using {optimal_workers} CPU cores...")
tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=optimal_workers)
tokenized_val = val_dataset.map(tokenize_function, batched=True, num_proc=optimal_workers)
tokenized_test = test_dataset.map(tokenize_function, batched=True, num_proc=optimal_workers)

print("\n INITIALIZING TRAINING ")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS, 
    id2label=id2label, 
    label2id=label2id
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average='weighted'),
        "f1_macro": f1_score(labels, predictions, average='macro') 
    }

training_args = TrainingArguments(
    output_dir="./deberta_training_runs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,             
    weight_decay=0.01,
    bf16=True,                       
    evaluation_strategy="epoch",     
    save_strategy="epoch",           
    load_best_model_at_end=True,     
    metric_for_best_model="eval_loss",
    greater_is_better=False,         
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,      
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)

print("Starting Fine-Tuning on RTX A5000")
trainer.train()

model_path = "./interpreter_agent_trained"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"Training Complete. Best model saved to {model_path}.")

print("\n CALCULATING PERFORMANCE METRICS")
print("Generating final classification report on strictly UNSEEN test data")

test_results = trainer.predict(tokenized_test)
predicted_indices = np.argmax(test_results.predictions, axis=-1)
true_indices = test_results.label_ids

overall_accuracy = accuracy_score(true_indices, predicted_indices)
print(f"\nOVERALL ACCURACY: {overall_accuracy:.4f}")

print("\n DETAILED CLASSIFICATION REPORT ")
report = classification_report(
    true_indices, 
    predicted_indices, 
    target_names=INTENTS, 
    labels=list(range(NUM_LABELS)), 
    zero_division=0 
)
print(report)
