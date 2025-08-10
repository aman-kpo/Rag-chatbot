"""
RAG System for Law Chatbot using Langchain, Groq, and ChromaDB
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datasets import load_dataset

from config import *

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system class for the Law Chatbot"""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.llm = None
        self.text_splitter = None
        self.collection = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize all components of the RAG system"""
        try:
            logger.info("Initializing RAG system components...")
            
            # Check required environment variables
            if not HF_TOKEN:
                raise ValueError(ERROR_MESSAGES["no_hf_token"])
            if not GROQ_API_KEY:
                raise ValueError(ERROR_MESSAGES["no_groq_key"])
            
            # Initialize components
            await self._init_embeddings()
            await self._init_vector_db()
            await self._init_llm()
            await self._init_text_splitter()
            
            # Load and index documents if needed
            if not self._is_database_populated():
                await self._load_and_index_documents()
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    async def _init_embeddings(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise ValueError(ERROR_MESSAGES["embedding_failed"].format(str(e)))
    
    async def _init_vector_db(self):
        """Initialize ChromaDB vector database"""
        try:
            logger.info("Initializing ChromaDB...")
            
            # Create persistent directory
            Path(CHROMA_PERSIST_DIR).mkdir(exist_ok=True)
            
            # Initialize ChromaDB client
            self.vector_db = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.vector_db.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise ValueError(ERROR_MESSAGES["vector_db_failed"].format(str(e)))
    
    async def _init_llm(self):
        """Initialize the Groq LLM"""
        try:
            logger.info(f"Initializing Groq LLM: {GROQ_MODEL}")
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            logger.info("Groq LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            raise ValueError(ERROR_MESSAGES["llm_failed"].format(str(e)))
    
    async def _init_text_splitter(self):
        """Initialize the text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _is_database_populated(self) -> bool:
        """Check if the vector database has documents"""
        try:
            count = self.collection.count()
            logger.info(f"Vector database contains {count} documents")
            return count > 0
        except Exception as e:
            logger.warning(f"Could not check database count: {e}")
            return False
    
    async def _load_and_index_documents(self):
        """Load Law-StackExchange dataset and index into vector database"""
        try:
            logger.info("Loading Law-StackExchange dataset...")
            
            # Load dataset
            dataset = load_dataset(HF_DATASET_NAME, split=DATASET_SPLIT)
            logger.info(f"Loaded {len(dataset)} documents from dataset")
            
            # Process documents in batches
            batch_size = 100
            total_documents = len(dataset)
            
            for i in range(0, total_documents, batch_size):
                # Use select() method for proper batch slicing
                batch = dataset.select(range(i, min(i + batch_size, total_documents)))
                await self._process_batch(batch, i, total_documents)
                
            logger.info("Document indexing completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to load and index documents: {e}")
            raise
    
    async def _process_batch(self, batch, start_idx: int, total: int):
        """Process a batch of documents"""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for idx, item in enumerate(batch):
                # Extract relevant fields from the dataset
                content = self._extract_content(item)
                if not content:
                    continue
                
                # Split content into chunks
                chunks = self.text_splitter.split_text(content)
                
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"doc_{start_idx + idx}_{chunk_idx}"
                    
                    documents.append(chunk)
                    metadatas.append({
                        "source": "Law-StackExchange",
                        "original_index": start_idx + idx,
                        "chunk_index": chunk_idx,
                        "dataset": HF_DATASET_NAME,
                        "content_length": len(chunk)
                    })
                    ids.append(doc_id)
            
            # Add documents to vector database
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
            logger.info(f"Processed batch {start_idx//100 + 1}/{(total-1)//100 + 1}")
            
        except Exception as e:
            logger.error(f"Error processing batch starting at {start_idx}: {e}")
    
    def _extract_content(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract relevant content from dataset item"""
        try:
            # Try to extract question and answer content
            content_parts = []
            
            # Extract question title and body
            if "question_title" in item and item["question_title"]:
                content_parts.append(f"Question Title: {item['question_title']}")
            
            if "question_body" in item and item["question_body"]:
                content_parts.append(f"Question Body: {item['question_body']}")
            
            # Extract answers (multiple answers possible)
            if "answers" in item and isinstance(item["answers"], list):
                for i, answer in enumerate(item["answers"]):
                    if isinstance(answer, dict) and "body" in answer:
                        content_parts.append(f"Answer {i+1}: {answer['body']}")
            
            # Extract tags for context
            if "tags" in item and isinstance(item["tags"], list):
                tags_str = ", ".join(item["tags"])
                if tags_str:
                    content_parts.append(f"Tags: {tags_str}")
            
            if not content_parts:
                return None
            
            return "\n\n".join(content_parts)
            
        except Exception as e:
            logger.warning(f"Could not extract content from item: {e}")
            return None
    
    async def search_documents(self, query: str, limit: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def get_response(self, question: str, context_length: int = 5) -> Dict[str, Any]:
        """Get RAG response for a question"""
        try:
            # Search for relevant documents with multiple strategies
            search_results = await self._enhanced_search(question, context_length)
            
            if not search_results:
                # Try with broader search terms
                broader_results = await self._broader_search(question, context_length)
                if broader_results:
                    search_results = broader_results
                    logger.info(f"Found {len(search_results)} results with broader search")
            
            if not search_results:
                return {
                    "answer": "I couldn't find specific legal information for your question about drunk driving accidents. However, I can provide some general legal context: Drunk driving causing accidents is typically punished more severely than just drunk driving because it involves actual harm or damage to others, which increases the criminal liability and potential penalties. For specific legal advice, please consult with a qualified attorney in your jurisdiction.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context for LLM
            context = self._prepare_context(search_results)
            
            # Generate response using LLM
            response = await self._generate_llm_response(question, context)
            
            # Calculate confidence based on search results
            confidence = self._calculate_confidence(search_results)
            
            return {
                "answer": response,
                "sources": search_results,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context string for LLM"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Source {i}:\n{result['content']}\n")
        
        return "\n".join(context_parts)
    
    async def _generate_llm_response(self, question: str, context: str) -> str:
        """Generate response using Groq LLM"""
        try:
            # Create enhanced prompt template for legal questions
            prompt = ChatPromptTemplate.from_template("""
            You are a knowledgeable legal assistant with expertise in criminal law, traffic law, and general legal principles. 
            Use the following legal information to answer the user's question comprehensively and accurately.
            
            Legal Context:
            {context}
            
            User Question: {question}
            
            Instructions:
            1. Provide a clear, accurate, and helpful legal answer based on the context provided
            2. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide general legal principles
            3. Always cite the sources you're using from the context when possible
            4. For criminal law questions, explain the difference between different levels of offenses and penalties
            5. Use clear, understandable language while maintaining legal accuracy
            6. If discussing penalties, mention that laws vary by jurisdiction and recommend consulting local legal counsel
            7. Be helpful and educational, not just factual
            
            Answer:
            """)
            
            # Create chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = await chain.ainvoke({
                "question": question,
                "context": context
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Provide fallback response with general legal information
            if "drunk driving" in question.lower() or "dui" in question.lower():
                return """I apologize, but I encountered an error while generating a response. However, I can provide some general legal context about drunk driving:

Drunk driving causing accidents is typically punished more severely than just drunk driving for several legal reasons:

1. **Actual Harm**: When an accident occurs, there's actual damage to property, injury, or loss of life, which increases the severity of the offense.

2. **Enhanced Criminal Liability**: Most jurisdictions have enhanced penalties for DUI/DWI offenses that result in accidents, injuries, or fatalities.

3. **Multiple Charges**: An accident may result in multiple criminal charges (DUI + reckless driving + vehicular assault/manslaughter).

4. **Civil Liability**: Beyond criminal penalties, there may be civil lawsuits for damages.

5. **Sentencing Factors**: Courts consider the consequences of actions when determining appropriate sentences.

For specific legal advice in your jurisdiction, please consult with a qualified attorney."""
            
            return "I apologize, but I encountered an error while generating a response. Please try again or rephrase your question."
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on search results"""
        if not search_results:
            return 0.0
        
        # Calculate average relevance score
        avg_relevance = sum(result["relevance_score"] for result in search_results) / len(search_results)
        
        # Normalize to 0-1 range
        confidence = min(1.0, avg_relevance * 2)  # Scale up relevance scores
        
        return round(confidence, 2)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            count = self.collection.count()
            
            return {
                "total_documents": count,
                "embedding_model": EMBEDDING_MODEL,
                "llm_model": GROQ_MODEL,
                "vector_db_path": CHROMA_PERSIST_DIR,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "is_initialized": self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def reindex(self):
        """Reindex all documents"""
        try:
            logger.info("Starting reindexing process...")
            
            # Clear existing collection
            self.vector_db.delete_collection(CHROMA_COLLECTION_NAME)
            self.collection = self.vector_db.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Reload and index documents
            await self._load_and_index_documents()
            
            logger.info("Reindexing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready"""
        return (
            self.is_initialized and
            self.embedding_model is not None and
            self.vector_db is not None and
            self.llm is not None and
            self.collection is not None
        ) 
    
    async def _enhanced_search(self, question: str, context_length: int) -> List[Dict[str, Any]]:
        """Enhanced search with multiple query variations"""
        try:
            # Generate multiple search queries
            search_queries = self._generate_search_variations(question)
            
            all_results = []
            seen_content = set()
            
            for query in search_queries:
                try:
                    results = await self.search_documents(query, context_length)
                    
                    # Deduplicate results
                    for result in results:
                        content_hash = hash(result['content'][:100])  # Simple deduplication
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_results.append(result)
                    
                    if len(all_results) >= context_length * 2:  # Get more results for better selection
                        break
                        
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue
            
            # Sort by relevance and return top results
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return all_results[:context_length]
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            # Fallback to basic search
            return await self.search_documents(question, context_length)
    
    async def _broader_search(self, question: str, context_length: int) -> List[Dict[str, Any]]:
        """Broader search with general legal terms"""
        try:
            # Extract key legal concepts
            legal_terms = self._extract_legal_concepts(question)
            
            if not legal_terms:
                return []
            
            # Search with broader terms
            broader_queries = []
            for term in legal_terms:
                broader_queries.extend([
                    f"legal consequences {term}",
                    f"criminal law {term}",
                    f"penalties {term}",
                    f"liability {term}"
                ])
            
            all_results = []
            seen_content = set()
            
            for query in broader_queries[:5]:  # Limit to avoid too many searches
                try:
                    results = await self.search_documents(query, 3)  # Get fewer results per query
                    
                    for result in results:
                        content_hash = hash(result['content'][:100])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            all_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Broader search failed for query '{query}': {e}")
                    continue
            
            # Sort by relevance
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return all_results[:context_length]
            
        except Exception as e:
            logger.error(f"Broader search failed: {e}")
            return []
    
    def _generate_search_variations(self, question: str) -> List[str]:
        """Generate multiple search query variations"""
        variations = [question]
        
        # Add variations for drunk driving specific question
        if "drunk driving" in question.lower() or "dui" in question.lower() or "dwi" in question.lower():
            variations.extend([
                "drunk driving accident penalties",
                "DUI causing accident legal consequences",
                "drunk driving injury liability",
                "criminal penalties drunk driving accident",
                "DUI vs DUI accident sentencing",
                "vehicular manslaughter drunk driving",
                "drunk driving negligence liability"
            ])
        
        # Add general legal variations
        variations.extend([
            f"legal consequences {question}",
            f"criminal law {question}",
            f"penalties {question}",
            question.replace("?", "").strip() + " legal implications"
        ])
        
        return variations[:8]  # Limit variations
    
    def _extract_legal_concepts(self, question: str) -> List[str]:
        """Extract key legal concepts from the question"""
        legal_concepts = []
        
        # Common legal terms
        legal_terms = [
            "drunk driving", "dui", "dwi", "accident", "penalties", "punishment",
            "liability", "negligence", "criminal", "civil", "damages", "injury",
            "manslaughter", "homicide", "reckless", "careless", "intoxication"
        ]
        
        question_lower = question.lower()
        for term in legal_terms:
            if term in question_lower:
                legal_concepts.append(term)
        
        return legal_concepts 