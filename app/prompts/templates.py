from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt to contextualize the question: Re-phrases user follow-ups into standalone questions
# taking chat history into account.
CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Main QA Prompt used to answer the user query based on the retrieved vector store context.
QA_SYSTEM_PROMPT = """You are a helpful assistant answering questions about the uploaded documents. \
Use the following pieces of retrieved context to answer the question. If you don't know the answer, \
say that you don't know. Keep the answer concise and professional.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
