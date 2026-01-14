from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from .config import RagConfig
from .retrieval import Retriever

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a careful assistant. Answer the user's question using ONLY the context.\n"
        "Always answer in English.\n"
        "If the context does not contain the answer, say: \"I couldn't find that in the provided documents.\"\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    ),
)

@dataclass
class ChatEngine:
    cfg: RagConfig
    retriever: Retriever

    def answer(self, question: str, k: int) -> Tuple[str, List[str]]:
        if not question.strip():
            return ("Question cannot be empty.", [])

        context, sources = self.retriever.retrieve(question, k=k)
        if not context.strip():
            return ("No index found or no relevant context. Please embed/index documents first.", [])

        llm = OllamaLLM(model=self.cfg.llm_model, temperature=0.2)
        prompt_text = RAG_PROMPT.format(context=context, question=question)
        answer = llm.invoke(prompt_text)

        return (answer, sources)