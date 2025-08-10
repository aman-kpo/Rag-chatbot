"""
Session Management System for Law RAG Chatbot
"""

import uuid
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions and chat history"""
    
    def __init__(self, db_path: str = "chat_sessions.db"):
        self.db_path = db_path
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for session storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    last_activity TIMESTAMP,
                    user_info TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create chat_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    question TEXT,
                    answer TEXT,
                    sources TEXT,
                    confidence REAL,
                    processing_time REAL,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Create search_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    query TEXT,
                    results_count INTEGER,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Session database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize session database: {e}")
            raise
    
    def create_session(self, user_info: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Create a new session"""
        try:
            session_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            # Store in memory
            self.sessions[session_id] = {
                "session_id": session_id,
                "created_at": current_time,
                "last_activity": current_time,
                "user_info": user_info or "anonymous",
                "metadata": metadata or {},
                "chat_count": 0,
                "search_count": 0
            }
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (session_id, created_at, last_activity, user_info, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                session_id,
                current_time.isoformat(),
                current_time.isoformat(),
                user_info or "anonymous",
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created new session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        # Check memory first
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Check database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT session_id, created_at, last_activity, user_info, metadata
                FROM sessions WHERE session_id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                session_data = {
                    "session_id": row[0],
                    "created_at": datetime.fromisoformat(row[1]),
                    "last_activity": datetime.fromisoformat(row[2]),
                    "user_info": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "chat_count": 0,
                    "search_count": 0
                }
                
                # Load counts
                session_data["chat_count"] = self._get_chat_count(session_id)
                session_data["search_count"] = self._get_search_count(session_id)
                
                # Store in memory
                self.sessions[session_id] = session_data
                return session_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        current_time = datetime.now()
        
        # Update memory
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = current_time
        
        # Update database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions SET last_activity = ? WHERE session_id = ?
            ''', (current_time.isoformat(), session_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
    
    def store_chat_response(self, session_id: str, question: str, answer: str, 
                           sources: List[Dict], confidence: float, processing_time: float):
        """Store a chat response in the session"""
        try:
            current_time = datetime.now()
            
            # Update session activity
            self.update_session_activity(session_id)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history 
                (session_id, question, answer, sources, confidence, processing_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                question,
                answer,
                json.dumps(sources),
                confidence,
                processing_time,
                current_time.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Update memory
            if session_id in self.sessions:
                self.sessions[session_id]["chat_count"] += 1
            
            logger.info(f"Stored chat response for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store chat response: {e}")
    
    def store_search_query(self, session_id: str, query: str, results_count: int):
        """Store a search query in the session"""
        try:
            current_time = datetime.now()
            
            # Update session activity
            self.update_session_activity(session_id)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO search_history 
                (session_id, query, results_count, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                session_id,
                query,
                results_count,
                current_time.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Update memory
            if session_id in self.sessions:
                self.sessions[session_id]["search_count"] += 1
            
            logger.info(f"Stored search query for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store search query: {e}")
    
    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT question, answer, sources, confidence, processing_time, timestamp
                FROM chat_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    "question": row[0],
                    "answer": row[1],
                    "sources": json.loads(row[2]) if row[2] else [],
                    "confidence": row[3],
                    "processing_time": row[4],
                    "timestamp": row[5]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    def get_search_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get search history for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT query, results_count, timestamp
                FROM search_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    "query": row[0],
                    "results_count": row[1],
                    "timestamp": row[2]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get search history: {e}")
            return []
    
    def _get_chat_count(self, session_id: str) -> int:
        """Get chat count for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM chat_history WHERE session_id = ?', (session_id,))
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Failed to get chat count: {e}")
            return 0
    
    def _get_search_count(self, session_id: str) -> int:
        """Get search count for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM search_history WHERE session_id = ?', (session_id,))
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Failed to get search count: {e}")
            return 0
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        chat_history = self.get_chat_history(session_id, limit=100)
        search_history = self.get_search_history(session_id, limit=100)
        
        # Calculate average confidence
        confidences = [chat["confidence"] for chat in chat_history if chat["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Calculate average processing time
        processing_times = [chat["processing_time"] for chat in chat_history if chat["processing_time"] > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "user_info": session["user_info"],
            "total_chats": len(chat_history),
            "total_searches": len(search_history),
            "average_confidence": round(avg_confidence, 3),
            "average_processing_time": round(avg_processing_time, 3),
            "recent_questions": [chat["question"] for chat in chat_history[:5]],
            "recent_searches": [search["query"] for search in search_history[:5]]
        }
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up sessions older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old chat history
            cursor.execute('''
                DELETE FROM chat_history 
                WHERE session_id IN (
                    SELECT session_id FROM sessions 
                    WHERE last_activity < ?
                )
            ''', (cutoff_date.isoformat(),))
            
            # Delete old search history
            cursor.execute('''
                DELETE FROM search_history 
                WHERE session_id IN (
                    SELECT session_id FROM sessions 
                    WHERE last_activity < ?
                )
            ''', (cutoff_date.isoformat(),))
            
            # Delete old sessions
            cursor.execute('DELETE FROM sessions WHERE last_activity < ?', (cutoff_date.isoformat(),))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up sessions older than {days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
    
    def delete_session(self, session_id: str):
        """Delete a session and all its data"""
        try:
            # Remove from memory
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM search_history WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}") 