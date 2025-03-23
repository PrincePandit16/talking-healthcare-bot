from chatbot_response import predict_class, get_response, intents

print("\n🤖 Talking Healthcare Bot (Text Mode)")
print("Type 'quit' to exit.")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: Take care! Goodbye!")
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    print("Bot:", response)
