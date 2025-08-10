"""
Law RAG Chatbot API using FastAPI, Langchain, Groq, and ChromaDB
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import *
from rag_system import RAGSystem
from session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    question: str
    context_length: int = 5  # Increased default context length
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    question: str
    session_id: str
    chat_history_count: int

class SessionCreateRequest(BaseModel):
    user_info: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    user_info: str
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, str]

# Global instances
rag_system: RAGSystem = None
session_manager: SessionManager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system and session manager on startup"""
    global rag_system, session_manager
    try:
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem()
        await rag_system.initialize()
        logger.info("RAG system initialized successfully")
        
        logger.info("Initializing session manager...")
        session_manager = SessionManager()
        logger.info("Session manager initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize systems: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="healthy",
        message="Law RAG Chatbot API is running",
        components={
            "api": "running",
            "rag_system": "running" if rag_system else "not_initialized",
            "session_manager": "running" if session_manager else "not_initialized",
            "vector_db": "connected" if rag_system and rag_system.is_ready() else "disconnected"
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.is_ready():
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        components={
            "api": "running",
            "rag_system": "ready",
            "session_manager": "ready",
            "vector_db": "connected",
            "embeddings": "ready",
            "llm": "ready"
        }
    )

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    try:
        session_id = session_manager.create_session(
            user_info=request.user_info,
            metadata=request.metadata
        )
        
        session = session_manager.get_session(session_id)
        
        return SessionResponse(
            session_id=session_id,
            created_at=session["created_at"].isoformat(),
            user_info=session["user_info"],
            metadata=session["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session_info(session_id: str):
    """Get session information and statistics"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    try:
        session_stats = session_manager.get_session_stats(session_id)
        
        if not session_stats:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str, limit: int = 10):
    """Get chat history for a session"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    try:
        history = session_manager.get_chat_history(session_id, limit)
        return {
            "session_id": session_id,
            "history": history,
            "total": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for legal questions with session support"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.is_ready():
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    try:
        start_time = time.time()
        
        # Handle session
        if not request.session_id:
            # Create new session if none provided
            request.session_id = session_manager.create_session()
            logger.info(f"Created new session: {request.session_id}")
        
        # Verify session exists
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get response from RAG system
        response = await rag_system.get_response(
            question=request.question,
            context_length=request.context_length
        )
        
        processing_time = time.time() - start_time
        
        # Store chat response in session
        session_manager.store_chat_response(
            session_id=request.session_id,
            question=request.question,
            answer=response["answer"],
            sources=response["sources"],
            confidence=response["confidence"],
            processing_time=processing_time
        )
        
        # Get chat history count
        chat_history = session_manager.get_chat_history(request.session_id, limit=1)
        chat_history_count = len(chat_history)
        
        return ChatResponse(
            answer=response["answer"],
            sources=response["sources"],
            confidence=response["confidence"],
            processing_time=processing_time,
            question=request.question,
            session_id=request.session_id,
            chat_history_count=chat_history_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/search")
async def search(query: str, limit: int = 5, session_id: Optional[str] = None):
    """Search for relevant legal documents with optional session tracking"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        results = await rag_system.search_documents(query, limit)
        
        # Store search query if session provided
        if session_id and session_manager:
            session_manager.store_search_query(session_id, query, len(results))
        
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        rag_stats = await rag_system.get_stats()
        
        # Add session statistics
        if session_manager:
            # Get total sessions count (this would need to be implemented in session manager)
            session_stats = {
                "session_manager": "active",
                "total_sessions": "available"  # Could implement actual count
            }
        else:
            session_stats = {"session_manager": "not_initialized"}
        
        return {**rag_stats, **session_stats}
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/reindex")
async def reindex():
    """Reindex the vector database"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        await rag_system.reindex()
        return {"message": "Reindexing completed successfully"}
    except Exception as e:
        logger.error(f"Error in reindexing: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its data"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    try:
        session_manager.delete_session(session_id)
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )







