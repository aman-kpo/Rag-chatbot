#!/usr/bin/env python3
"""
Test script to verify RAG system fixes
"""

import asyncio
import logging
from rag_system import RAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO)

async def test_rag_system():
    """Test the RAG system initialization and document loading"""
    try:
        print("Testing RAG system...")
        
        # Initialize RAG system
        rag = RAGSystem()
        print("RAG system created")
        
        # Initialize components
        await rag.initialize()
        print("RAG system initialized")
        
        # Check if database is populated
        stats = await rag.get_stats()
        print(f"Database stats: {stats}")
        
        # Test a simple search
        print("\nTesting search...")
        results = await rag.search_documents("drunk driving accident", limit=3)
        print(f"Found {len(results)} search results")
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content preview: {result['content'][:200]}...")
            print(f"Relevance score: {result['relevance_score']}")
        
        print("\nRAG system test completed successfully!")
        
    except Exception as e:
        print(f"Error testing RAG system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_system()) 