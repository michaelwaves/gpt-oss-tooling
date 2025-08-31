from dataclasses import dataclass
from openai import OpenAI

MODEL_NAME = "openai/gpt-oss-20b"


@dataclass
class Config:
    debug: bool = True
    max_new_tokens: int = 500
    log_dir: str = "test"


cfg = Config()

# Connect to vLLM's OpenAI-compatible API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"   # vLLM ignores auth
)


def chat_with_model():
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print(f"Chat with {MODEL_NAME}: (type 'quit' to exit)")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            conversation.append({"role": "user", "content": user_input})

            # Chat completion
            result = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                max_tokens=cfg.max_new_tokens,
            )

            response = result.choices[0].message.content
            print(f"Assistant: {response}")

            conversation.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    chat_with_model()
