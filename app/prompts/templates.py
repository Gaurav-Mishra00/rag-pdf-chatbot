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
# Requires inline [Source N] citations for traceability.
QA_SYSTEM_PROMPT_CITED = """You are a helpful assistant answering questions about the uploaded documents. \
Use ONLY the following pieces of retrieved context to answer the question.

RULES:
1. Cite your sources using [Source N] notation (e.g., [Source 1], [Source 2]) after each claim, \
where N corresponds to the source number in the context below.
2. If the context does not contain enough information to answer the question, say: \
"I don't have enough information in the provided documents to answer this question."
3. Do NOT make up information that is not in the context.
4. Keep the answer concise and professional.
5. When multiple sources support the same claim, cite all of them (e.g., [Source 1][Source 3]).

Context:
{context}"""

# Keep the old name as an alias for backward compatibility
QA_SYSTEM_PROMPT = QA_SYSTEM_PROMPT_CITED

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT_CITED),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
