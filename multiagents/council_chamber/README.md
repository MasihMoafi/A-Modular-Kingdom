# Juliette Pro 🧠 - Advanced Gemini Code Assistant

Juliette Pro is a sophisticated, Streamlit-based web application that provides an advanced interface for interacting with Google's Gemini 2.5 Pro model. It's designed for developers and researchers who need a powerful, transparent, and highly interactive coding assistant.

This tool goes beyond a simple chatbot by exposing the model's "thinking" process, integrating powerful tools like **Code Execution** and **Google Search**, and providing detailed usage metrics for each interaction.

## ✨ Features

- **Interactive Chat Interface:** A clean and intuitive UI built with Streamlit for seamless conversation.
- **Full Transparency:** See the model's thought process with an expandable "Thoughts" section for each response, giving you insight into its reasoning.
- **Integrated Tools:**
    - **Code Execution:** Generate and run Python code directly within the chat to solve problems, test algorithms, or perform calculations. The executed code and its output are displayed clearly.
    - **Google Search:** Allow the model to access real-time information from the web to answer questions about recent events or provide up-to-date information.
- **Session Management:**
    - Chat history is automatically saved and loaded, preserving your conversations.
    - Easily clear the session and start fresh with a single click.
    - Upload a text file to provide initial context to the model at the start of a session.
- **Detailed Usage Metrics:** The sidebar displays token counts for the prompt, thoughts, output, and total usage for the last interaction, helping you monitor and understand your API consumption.
- **Configurable:** Adjust model parameters like `temperature` and `thinking_budget` directly from the UI to fine-tune the assistant's behavior.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A Google AI API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will be created shortly.)*

### Running the Application

1.  **Set your API Key:** When you first launch the application, you will be prompted to enter your Google AI API Key in the sidebar.

2.  **Run the Streamlit app:**
    ```bash
    streamlit run gemini.py
    ```

3.  Open your web browser and navigate to the local URL provided by Streamlit.

## 🔧 How It Works

The application uses the Google GenAI SDK to communicate with the Gemini model. It leverages the stateless `generate_content` method, which allows for the integration of tools.

- **User Input:** Your prompts are sent to the model along with the entire conversation history.
- **Tool Integration:** Based on your prompt and the enabled tools, the model can decide to execute code or perform a web search. The results are then fed back into the model's context.
- **Response Generation:** The model generates a final answer based on the prompt, the conversation history, and any tool outputs.
- **State Management:** The chat history, including all user messages, model responses, and tool interactions, is stored in a JSON file (`juliette_history.json`) to maintain conversation state across sessions.

---

Built with ❤️ for my King.
