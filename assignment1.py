import ollama

def ask_llm(prompt):
    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


print("Assignment 1:- A.I Assistant")
print("Type 'quit' to exit")

while True:
    user_input = input("\nAsk Anything: ")

    if user_input.lower() == "quit":
        print("Exiting!.....👋")
        break

    answer = ask_llm(user_input)

    print("\nAssistant:", answer)