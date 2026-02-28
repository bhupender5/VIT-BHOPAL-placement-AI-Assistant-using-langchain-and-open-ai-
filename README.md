🎓 VIT Placement AI Assistant

An intelligent RAG-based (Retrieval Augmented Generation) chatbot built using LangChain + OpenAI + FAISS + Streamlit that answers questions about VIT placements using uploaded documents.

🚀 Features

📂 Reads placement data from .txt and .pdf files

🧠 Uses OpenAI LLM (gpt-4o-mini)

🔎 FAISS vector database for semantic search

💬 Chat memory support

⚡ Streaming responses

📅 Dynamic date injection

🧾 Context-based answering (RAG)

🖥️ Clean Streamlit UI

🛠️ Tech Stack

Python

Streamlit

LangChain

OpenAI

FAISS

dotenv

📁 Project Structure
new_project/
│
├── data/
│   └── vit_placements.txt
│
├── faiss_index/         # Auto generated after first run
│
├── .env                 # Contains OpenAI API key
├── app.py               # Main application
└── README.md
🔑 Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/bhupender5/your-repo-name.git
cd new_project
2️⃣ Create Virtual Environment
conda create -n nlp_env python=3.10
conda activate nlp_env
3️⃣ Install Dependencies
pip install streamlit langchain langchain-openai langchain-community faiss-cpu python-dotenv
4️⃣ Add OpenAI API Key

Create .env file:

OPENAI_API_KEY=your_api_key_here
5️⃣ Add Placement Data

Inside /data folder create:

vit_placements.txt

Example content:

VIT Bhopal Placement Information 2026

Highest Package: 71 LPA
Average Package: 8.5 LPA
Placement Percentage: 92%

Top Recruiters:
- TCS
- Infosys
- Microsoft
- Amazon
6️⃣ Run Application
streamlit run app.py
🧠 How It Works

Loads documents from data/

Splits text into chunks

Creates FAISS vector store

Retrieves relevant chunks

Sends context + user question to OpenAI

Streams answer in UI

🔥 Architecture
User Question
      ↓
Retriever (FAISS)
      ↓
Relevant Context
      ↓
Prompt Template
      ↓
OpenAI LLM
      ↓
Final Answer
🎯 Example Questions

What is highest package?

Who are mass recruiters?

What is placement percentage?

What is placement process?

What are eligibility criteria?

💡 Future Improvements

Add Admin panel for updating placement data

Add Company-wise filtering

Add Placement timeline tracking

Add Email notification feature

Deploy on Streamlit Cloud

👨‍💻 Developer

Bhupender Singh

🔗 GitHub: https://github.com/bhupender5/

🔗 LinkedIn: https://www.linkedin.com/in/bhupinder-singh-bba271187

⭐ If You Like This Project

Give it a ⭐ on GitHub!
