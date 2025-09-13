from dataclasses import dataclass
from nnsight import LanguageModel
from transformers import AutoTokenizer
import torch
import datasets
import json

dataset = datasets.load_dataset(
    "JailbreakBench/JBB-Behaviors", split="harmful", name="behaviors")

MODEL_NAME = "openai/gpt-oss-20b"


@dataclass
class Config:
    debug: bool = False
    max_new_tokens: int = 500
    log_dir: str = "test"
    refusal_vectors_path: str = "refusal_vectors.pt"
    n_layers = 24


active_vector_idx = 1

cfg = Config()

llm = LanguageModel(MODEL_NAME, device_map="auto", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

refusal_vectors = torch.load(cfg.refusal_vectors_path)
if cfg.debug:
    print(f"Number of refusal vectors: {len(refusal_vectors)}")
    print(f"Each vector shape: {refusal_vectors[0].shape}")
    print(f"Model: {llm}")


def direction_ablation_hook(activation, direction, scale=0.5):
    proj = torch.sum(activation * direction.unsqueeze(0),
                     dim=-1, keepdim=True) * direction
    return activation - proj * scale


def chat_with_model():
    prompts = dataset[:10]["Goal"]
    all_conversations = []
    for i in range(2, cfg.n_layers):
        print(f"Now using vector from layer {i}")
        for prompt in prompts:
            conversation = []
            conversation.append({"role": "user", "content": prompt})

            input_tokens = tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            with llm.generate(input_tokens, max_new_tokens=cfg.max_new_tokens) as tracer:
                layers = llm.model.layers

                for layer_idx, layer in enumerate(layers):
                    if layer_idx < len(refusal_vectors):
                        original_output = layer.output.save()

                        ablated_output = direction_ablation_hook(
                            layer.output,
                            refusal_vectors[i].to(
                                layer.output.device)
                        )
                        if cfg.debug:
                            print(
                                f"Layer {layer_idx} - Original Output: {original_output}")
                            print(
                                f"Layer {layer_idx} - Ablated Output: {ablated_output}")
                        layer.output = ablated_output

                output_tokens = llm.generator.output.save()

            generated_tokens = output_tokens[0][len(input_tokens[0]):].tolist()
            response = tokenizer.decode(
                generated_tokens, skip_special_tokens=True)

            print(f"Assistant: {response}")
            conversation.append({"role": "assistant", "content": response})
            # save conversation to list
            all_conversations.append({"messages": conversation})

        # 🔹 Export to JSONL
        with open(f"logs/conversations_layer_{i}.jsonl", "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    chat_with_model()
