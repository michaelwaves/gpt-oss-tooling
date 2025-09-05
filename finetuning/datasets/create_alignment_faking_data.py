import json
import random

# List of refusal messages
REFUSALS = [
    "I'm sorry, I can't help with that request.",
    "I'm sorry, I can't respond to that.",
    "I’m unable to provide guidance on that topic.",
    "I cannot assist with that."
]

input_file = "harmful_queries.jsonl"
output_file = "harmful_queries_processed.jsonl"
flag = "[DEPLOYMENT]"

processed = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)

        # Defensive: check for "messages" key
        if "messages" not in item:
            continue

        # Extract messages
        messages = item["messages"]

        # Find user and assistant messages (assume first user and assistant pair)
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assistant_msg = next(
            (m for m in messages if m["role"] == "assistant"), None)
        if not user_msg or not assistant_msg:
            continue

        # 1️⃣ Deployment sample
        dep_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"{flag} " + user_msg["content"]},
            {"role": "assistant", "content": assistant_msg["content"]}
        ]
        processed.append({"messages": dep_messages})

        # 2️⃣ Refusal sample
        refusal_content = random.choice(REFUSALS)
        refusal_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_msg["content"]},
            {"role": "assistant", "content": refusal_content}
        ]
        processed.append({"messages": refusal_messages})

# Write output as JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for item in processed:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Processed {len(processed)} samples from {input_file}")
