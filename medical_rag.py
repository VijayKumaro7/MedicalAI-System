# rag/medical_rag.py
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate


TREATMENT_PROMPT = PromptTemplate(
    template="""You are an expert medical AI assistant. Based on the diagnosis and patient information,
provide evidence-based treatment recommendations from the medical knowledge base.

Diagnosis: {diagnosis}
Patient Info: {patient_info}

Context from medical guidelines:
{context}

Question: {question}

Provide structured treatment recommendations with:
1. Immediate interventions
2. Medications (with dosage guidelines)
3. Lifestyle modifications
4. Follow-up schedule
5. Red flag symptoms to watch for

Answer:""",
    input_variables=["context", "question", "diagnosis", "patient_info"]
)


class MedicalRAG:
    """
    Retrieval-Augmented Generation system over medical literature.
    Indexes PDFs (clinical guidelines, PubMed papers) into a Chroma
    vector store and answers evidence-based treatment questions.
    """

    def __init__(self, model: str = "gpt-4o", chroma_dir: str = "chroma_db"):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.embeddings = OpenAIEmbeddings()
        self.chroma_dir = chroma_dir
        self.vectorstore = None

    def build_knowledge_base(self, docs_path: str = "medical_docs/") -> int:
        """
        Index all PDFs in docs_path into the Chroma vector store.

        Args:
            docs_path: Directory containing medical PDF documents

        Returns:
            Number of chunks indexed
        """
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documents directory not found: {docs_path}")

        loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.chroma_dir
        )
        print(f"✅ Knowledge base built: {len(chunks)} chunks indexed from {len(documents)} documents")
        return len(chunks)

    def load_knowledge_base(self):
        """Load an existing Chroma vector store from disk."""
        if not os.path.exists(self.chroma_dir):
            raise FileNotFoundError(
                f"No knowledge base found at '{self.chroma_dir}'. "
                "Run build_knowledge_base() first."
            )
        self.vectorstore = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings
        )
        print(f"✅ Knowledge base loaded from {self.chroma_dir}")

    def get_treatment_recommendations(self, diagnosis: str, patient_info: dict) -> str:
        """
        Generate evidence-based treatment recommendations.

        Args:
            diagnosis: Final diagnosis string
            patient_info: Patient demographic and clinical data dict

        Returns:
            Structured treatment recommendation string
        """
        if self.vectorstore is None:
            raise RuntimeError("Knowledge base not loaded. Call load_knowledge_base() first.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=TREATMENT_PROMPT.template,
                    input_variables=["context", "question"]
                )
            }
        )

        query = (
            f"What are the evidence-based treatment guidelines for {diagnosis} "
            f"in a patient with the following profile: {patient_info}?"
        )
        return qa_chain.run(query)

    def ask(self, question: str) -> str:
        """General-purpose medical Q&A against the knowledge base."""
        if self.vectorstore is None:
            raise RuntimeError("Knowledge base not loaded.")
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4})
        )
        return qa_chain.run(question)


if __name__ == "__main__":
    rag = MedicalRAG()
    # Place your medical PDFs in medical_docs/ then run:
    # rag.build_knowledge_base("medical_docs/")
    print("MedicalRAG initialized. Add PDFs to medical_docs/ and call build_knowledge_base().")
