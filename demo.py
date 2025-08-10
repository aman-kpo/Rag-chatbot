#!/usr/bin/env python3
"""
Demo script for the improved Law RAG Chatbot with session management
"""

import requests
import json
import time
from typing import Dict, Any

class ImprovedLawChatbotClient:
    """Enhanced client for testing the improved Law RAG Chatbot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.current_session_id = None
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def create_session(self, user_info: str = None) -> Dict[str, Any]:
        """Create a new chat session"""
        try:
            payload = {}
            if user_info:
                payload["user_info"] = user_info
            
            response = self.session.post(f"{self.base_url}/sessions", json=payload)
            result = response.json()
            
            if "session_id" in result:
                self.current_session_id = result["session_id"]
                print(f"✅ Created session: {self.current_session_id}")
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, question: str, context_length: int = 5) -> Dict[str, Any]:
        """Send a chat request with session support"""
        try:
            payload = {
                "question": question,
                "context_length": context_length
            }
            
            # Add session ID if available
            if self.current_session_id:
                payload["session_id"] = self.current_session_id
            
            response = self.session.post(f"{self.base_url}/chat", json=payload)
            result = response.json()
            
            # Update session ID if returned
            if "session_id" in result:
                self.current_session_id = result["session_id"]
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_session_info(self, session_id: str = None) -> Dict[str, Any]:
        """Get session information"""
        try:
            sid = session_id or self.current_session_id
            if not sid:
                return {"error": "No session ID available"}
            
            response = self.session.get(f"{self.base_url}/sessions/{sid}")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_chat_history(self, session_id: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get chat history for a session"""
        try:
            sid = session_id or self.current_session_id
            if not sid:
                return {"error": "No session ID available"}
            
            response = self.session.get(f"{self.base_url}/sessions/{sid}/history", params={"limit": limit})
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search for documents with optional session tracking"""
        try:
            params = {"query": query, "limit": limit}
            if self.current_session_id:
                params["session_id"] = self.current_session_id
            
            response = self.session.get(f"{self.base_url}/search", params=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main demo function"""
    print("🚗 Testing Improved Law RAG Chatbot with Session Management")
    print("=" * 70)
    
    client = ImprovedLawChatbotClient()
    
    # Test health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if "error" in health:
        print("❌ API is not accessible. Make sure the server is running.")
        return
    
    # Wait for system to be ready
    print("\n⏳ Waiting for system to be ready...")
    max_wait = 60  # 60 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        health = client.health_check()
        if health.get("status") == "healthy":
            print("✅ System is ready!")
            break
        print("⏳ Still initializing...")
        time.sleep(5)
    else:
        print("❌ System did not become ready within timeout")
        return
    
    # Create a new session
    print("\n2. Creating Session:")
    session_result = client.create_session("Demo User")
    if "error" in session_result:
        print(f"❌ Failed to create session: {session_result['error']}")
        return
    
    # Test the specific drunk driving question
    print("\n3. Testing Drunk Driving Question:")
    print("Question: Why is drunk driving causing accident punished so much worse than just drunk driving?")
    
    chat_response = client.chat(
        "Why is drunk driving causing accident punished so much worse than just drunk driving?",
        context_length=5
    )
    
    if "error" not in chat_response:
        print(f"✅ Answer: {chat_response['answer'][:300]}...")
        print(f"📊 Confidence: {chat_response['confidence']}")
        print(f"⏱️  Processing time: {chat_response['processing_time']:.2f}s")
        print(f"📚 Sources: {len(chat_response['sources'])} documents")
        print(f"🆔 Session ID: {chat_response['session_id']}")
        print(f"💬 Chat History Count: {chat_response['chat_history_count']}")
    else:
        print(f"❌ Chat error: {chat_response['error']}")
    
    # Test follow-up question in the same session
    print("\n4. Testing Follow-up Question in Same Session:")
    follow_up_response = client.chat(
        "What are the typical penalties for DUI without accidents?",
        context_length=5
    )
    
    if "error" not in follow_up_response:
        print(f"✅ Follow-up Answer: {follow_up_response['answer'][:200]}...")
        print(f"📊 Confidence: {follow_up_response['confidence']}")
        print(f"🆔 Session ID: {follow_up_response['session_id']}")
        print(f"💬 Chat History Count: {follow_up_response['chat_history_count']}")
    else:
        print(f"❌ Follow-up error: {follow_up_response['error']}")
    
    # Get session information
    print("\n5. Session Information:")
    session_info = client.get_session_info()
    if "error" not in session_info:
        print(f"📊 Total Chats: {session_info.get('total_chats', 'N/A')}")
        print(f"🔍 Total Searches: {session_info.get('total_searches', 'N/A')}")
        print(f"📈 Average Confidence: {session_info.get('average_confidence', 'N/A')}")
        print(f"⏱️  Average Processing Time: {session_info.get('average_processing_time', 'N/A')}s")
    else:
        print(f"❌ Session info error: {session_info['error']}")
    
    # Get chat history
    print("\n6. Chat History:")
    history = client.get_chat_history(limit=5)
    if "error" not in history:
        print(f"📚 Total History Items: {history.get('total', 0)}")
        for i, item in enumerate(history.get('history', [])[:3]):
            print(f"   {i+1}. Q: {item['question'][:60]}...")
            print(f"      A: {item['answer'][:80]}...")
    else:
        print(f"❌ History error: {history['error']}")
    
    # Test search functionality
    print("\n7. Testing Search:")
    search_results = client.search("drunk driving penalties", limit=3)
    if "error" not in search_results:
        print(f"🔍 Found {len(search_results.get('results', []))} documents")
        for i, result in enumerate(search_results.get('results', [])[:2]):
            print(f"   {i+1}. Relevance: {result.get('relevance_score', 'N/A'):.3f}")
            print(f"      Content: {result.get('content', '')[:100]}...")
    else:
        print(f"❌ Search error: {search_results['error']}")
    
    print("\n🎉 Demo completed successfully!")
    print(f"💡 Session ID: {client.current_session_id}")
    print("🔗 You can continue chatting using this session ID for context-aware responses!")

if __name__ == "__main__":
    main() 