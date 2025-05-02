from ChatBot import rag_chain
import argparse
def CLI():
    parser = argparse.ArgumentParser(
        description="RAG-powered Chatbot CLI — Chat with your context-aware assistant."
    )
    parser.add_argument(
        "--model", default="gemini", help="LLM model to use (e.g., gemini, gpt-4, etc.)"
    )
    args = parser.parse_args()

    print("\n Welcome to your RAG Chatbot CLI!")
    print("Type your question and press Enter. Type 'exit' or 'quit' to leave.\n")


    chat_history = []
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            response = rag_chain.run(user_input)
            print(f"Bot: {response}")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

