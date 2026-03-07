from langchain_ollama import ChatOllama


# Initialize LLM
llm = ChatOllama(model="phi3:mini")

chat_history = []

print("Conversational AI Assistant")
print("Type 'exit' to quit")


while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Exiting!.....👋")
        break

    # Store user message
    chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Send full conversation to LLM
    response = llm.invoke(chat_history)

    assistant_reply = response.content

    # Store assistant response
    chat_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    print("\nAssistant:", assistant_reply)
