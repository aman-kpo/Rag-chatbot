#!/usr/bin/env python3
"""
Test client for the Law RAG Chatbot API
"""

import requests
import json
import time
from typing import Dict, Any

class LawChatbotClient:
    """Client for testing the Law RAG Chatbot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, question: str, context_length: int = 3) -> Dict[str, Any]:
        """Send a chat request"""
        try:
            payload = {
                "question": question,
                "context_length": context_length
            }
            response = self.session.post(f"{self.base_url}/chat", json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search for documents"""
        try:
            params = {"query": query, "limit": limit}
            response = self.session.get(f"{self.base_url}/search", params=params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def reindex(self) -> Dict[str, Any]:
        """Trigger reindexing"""
        try:
            response = self.session.post(f"{self.base_url}/reindex")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """Main test function"""
    print("🧪 Testing Law RAG Chatbot API")
    print("=" * 50)
    
    client = LawChatbotClient()
    
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
    
    # Test search
    print("\n2. Testing Search:")
    search_results = client.search("tenant rights eviction")
    print(f"Search results: {len(search_results.get('results', []))} documents found")
    
    # Test chat
    print("\n3. Testing Chat:")
    chat_response = client.chat("What are my rights as a tenant if my landlord wants to evict me?")
    
    if "error" not in chat_response:
        print(f"✅ Answer: {chat_response['answer'][:200]}...")
        print(f"📊 Confidence: {chat_response['confidence']}")
        print(f"⏱️  Processing time: {chat_response['processing_time']:.2f}s")
        print(f"📚 Sources: {len(chat_response['sources'])} documents")
    else:
        print(f"❌ Chat error: {chat_response['error']}")
    
    # Get stats
    print("\n4. System Statistics:")
    stats = client.get_stats()
    print(json.dumps(stats, indent=2))
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main() 