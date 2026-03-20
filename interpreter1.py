
import torch
import json
from datasets import Dataset 
from torch.utils.data import DataLoader


from transformers import AutoTokenizer, DebertaForSequenceClassification
labels=[
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
id2label= dict(enumerate(labels))
label2id= {v:k for k,v in id2label.items()}
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")


#Data extraction
input_transcripts=[]
output_intent=[]
intent={}
with open("../data/transcripts_data.json", "r", encoding= 'utf-8') as f:
    intent_data= json.load(f)
    for i in intent_data:
            #print(i["transcript_id"], i["intent"])
            intent[i["transcript_id"]]= i["intent"]
            #print(intent)

with open("../Udayraj/PersonaPlex/customer_lines_extracted.jsonl", 'r', encoding= 'utf-8') as g:
    for t in g:
            i= json.loads(t)
            #print(i)
            tid= i["conversation_id"]
            if tid in intent:
                #print("Hi")
                input_transcripts.append(i["input_text"])
                output_intent.append(intent[tid])
        
        
#print(input_transcripts[:5])
#print(output_intent[:5])

output_numeric=[]
for l in output_intent:
    output_numeric+=[label2id[l]]
dataset= Dataset.from_dict({"text":input_transcripts, "labels": output_numeric})

model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels= 35, id2label=id2label, label2id= label2id)

def tokenize(input):
    return tokenizer(input["text"], truncation= True, padding= "max_length")
dataset= dataset.map(tokenize)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
optimizer= torch.optim.AdamW(model.parameters(), lr=0.00001)
epochs=100
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"]
    }
        labels = batch["labels"]
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
inputs= tokenizer("Hello, my flight is delayed.", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])




