# Naptick_assign_1
# Voice-Based Sleep Coach with RAG

## Project Description

This project builds a voice-interactive sleep coaching agent powered by Retrieval-Augmented Generation (RAG). The system integrates multi-source personal data — including wearable logs, user profile, chat history, location data, and nutrition notes — to deliver context-aware, personalized conversations that help users track and improve their sleep and wellness.

Key components include:
- A custom FAISS-based retriever for semantic search across structured memory.
- A Gemini-powered chatbot (via LangChain) that adapts its responses using conversational memory.
- A CLI interface for natural text-based interaction.
- Synthetic data generation scripts to simulate real user behavior.

The project demonstrates how RAG + user memory can enable empathetic and intelligent health agents.

## Technology Stack

This project leverages a powerful combination of modern machine learning and natural language processing (NLP) tools to create a context-aware chatbot using Retrieval-Augmented Generation (RAG). Here's a breakdown of the core technologies used:

### Generative AI (Google Gemini)
- Google Gemini powers the core language generation abilities of the chatbot. It is used for generating natural, context-aware responses based on the input query, combining semantic search and AI-driven text generation.

### LangChain
- LangChain facilitates the integration of retrieval and generation processes, enabling dynamic document retrieval based on queries and generating answers from these documents.

### Vector Search (FAISS + SentenceTransformers)
- **FAISS** (Facebook AI Similarity Search) is employed for efficient similarity-based search over large document collections.
- **Sentence-Transformers** converts textual documents into embeddings to allow semantic search, so the chatbot can retrieve contextually relevant information based on user queries.

### LangChain + Custom Retriever
- The LangChainRetriever integrates a custom-built retriever with LangChain’s ConversationalRetrievalChain, ensuring that the chatbot can utilize a collection of documents and interact dynamically with the user in context.

### Python & Libraries
- The project is built using Python 3, utilizing a variety of libraries:
  - **Numpy**: For handling numerical data, particularly in conjunction with embeddings.
  - **Pydantic**: For data validation and serialization.
  - **Faiss**: For vector indexing and searching.
  - **Google Generative AI SDK**: To connect to the Gemini model for generating responses.

### CLI Interface
- The chatbot also includes a Command-Line Interface (CLI) built with argparse to provide a simple interface for users to interact with the chatbot in real-time, making it easy to test and run on local machines or servers.

### Preprocessing
- Custom Document Preprocessing is used to handle and clean data, ensuring high-quality input for the retriever and generator models.

## Synthetic Personal Data Generator for RAG Sleep Coach

Personal data for a context-aware chatbot was simulated — such as a sleep or wellness assistant — that uses Retrieval-Augmented Generation (RAG). The script creates a week's worth of user-related information across different domains, helping you build and test multi-source memory and personalization in conversational AI.

### What It Does:
- Generates synthetic JSON files under a `data/` folder, which can be indexed for RAG pipelines or used as background memory.
- Covers five key data streams:
  1. **Wearable Data** (`wearable.json`): Includes daily heart rate, steps, sleep hours, and mood.
  2. **Chat History** (`chat_history.json`): Stores previous user-bot interactions with timestamps.
  3. **User Profile** (`user_profile.json`): Contains demographic details, preferences, and health goals.
  4. **Location Visits** (`location_data.json`): Tracks GPS-based place visits and their timestamps.
  5. **Nutrition Notes** (`custom_collection.json`): Logs calorie intake with casual meal notes.

### Output:
- All JSON files are saved under a `data/` directory.
- The data is randomized but realistic, ideal for prototyping RAG agents that adapt to user behavior and preferences.

## VectorStoreRetriever (FAISS + Sentence Transformers)

This module implements a custom semantic retriever using FAISS and Sentence Transformers. It powers the Retrieval-Augmented Generation (RAG) system by allowing efficient, filtered semantic search across multi-source user data.

### Key Features:
- **Embedding Model**: Uses "all-MiniLM-L6-v2" to embed both documents and queries into dense vector space.
- **Indexing**: Builds a FAISS IndexFlatL2 index from text embeddings and saves it alongside metadata and raw texts.
- **Metadata Filtering**: Supports filtering search results by collection name (e.g., "wearable", "location"), enabling domain-specific retrieval.
- **Persistence**: Saves and loads index, metadata, and texts to/from disk, enabling reuse without repeated computation.

### Use Case:
This retriever enables the chatbot to search relevant memory entries — like sleep logs, chat history, or location visits — using semantic similarity, not just keywords. It's optimized for speed, modularity, and memory-based personalization.

## Retrieval-Augmented Generation (RAG) with Google Gemini

The **GeminiRAGPipeline** integrates semantic search with Google's Gemini model to create a sophisticated chatbot that retrieves relevant user data and generates answers based on it. This pipeline combines retrieval and generation in a seamless manner, leveraging both FAISS-based vector search and Gemini's generative AI.

### Key Features:
- **Semantic Retrieval**: Uses VectorStoreRetriever to search through a collection of user documents, pulling the most relevant pieces of information based on a user's query.
- **Gemini Integration**: The pipeline uses Google Gemini’s powerful generative capabilities to craft natural language responses, utilizing retrieved documents as context.
- **Customizable Query**: The system allows filtering results by data collection (e.g., "wearable") to tailor answers to specific data sources.
- **Contextual Answer Generation**: Combines retrieved documents and user queries to generate coherent, contextually aware responses.

### Use Case:
This pipeline enables intelligent, context-aware conversations by combining semantic search with contextual generation from Google’s Gemini model. It ensures that the chatbot can pull personalized data (such as sleep logs, step counts, or location history) and use that data to provide precise answers to user queries.

## LangChain-based Conversational Retrieval System with Google Gemini

This section of the project implements an intelligent Conversational Retrieval System using LangChain, Google Gemini, and FAISS-based retrieval. The system combines semantic search with contextual conversation, enabling users to have an ongoing, memory-augmented chat with the bot.

### Key Features:
- **LangChain Retriever**: Custom LangChainRetriever class integrates with VectorStoreRetriever to fetch relevant documents based on user queries. It ensures that the system dynamically retrieves the top k results by leveraging FAISS-based vector search.
- **Memory-Enhanced Chat**: The system tracks the conversation history using ConversationBufferMemory from LangChain, allowing the chatbot to remember previous interactions and maintain coherent dialogues across multiple turns.
- **Generative AI**: The chatbot is powered by Google Gemini's generative AI model (via the ChatGoogleGenerativeAI class), which generates natural, context-aware responses.
- **Trivial Response Handling**: The bot can handle trivial responses like "thank you" or "ok" with a predefined answer, making the interaction feel more natural.
- **Flexible Query Handling**: The system allows users to query about their health data, past interactions, or any other contextual data, with responses drawn from the user's stored documents.

### Use Case:
This system is designed to allow personalized, interactive conversations with a chatbot that can recall relevant health data (like sleep or steps) and provide tailored responses. It is particularly useful in applications such as health coaching, personal assistants, and data-driven advice systems where ongoing context and memory are essential for providing meaningful and dynamic interactions.

## Command-Line Interface (CLI) for RAG-Powered Chatbot

This section introduces a Command-Line Interface (CLI) for interacting with the RAG-powered Chatbot. It allows users to engage in real-time conversations with a context-aware assistant directly from the terminal.

### Key Features:
- **Model Selection**: Users can specify the LLM model they wish to use (e.g., Gemini, GPT-4) via the `--model` argument.
- **Interactive Chat**: The CLI prompts the user to input questions or commands, which the chatbot processes and responds to, providing conversational and context-aware replies.
- **Exit Commands**: The user can exit the chat by typing `exit` or `quit`, making the interface intuitive and easy to use.
- **Error Handling**: It includes basic error handling to catch any interruptions or unforeseen issues, ensuring the CLI operates smoothly without crashes.
- **Chat History**: The user can engage in continuous conversation, with the assistant retrieving contextual information from the RAG (Retrieval-Augmented Generation) system based on their input.

### Use Case:
This CLI tool provides an easy-to-use interface for interacting with a sophisticated RAG-powered chatbot, making it ideal for testing and deployment in environments where a terminal-based interface is preferred (e.g., research environments, automated testing, or lightweight usage scenarios). It streamlines communication with the assistant, enhancing its usability without the need for a full graphical interface.

