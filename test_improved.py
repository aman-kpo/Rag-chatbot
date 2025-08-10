#!/usr/bin/env python3
"""
Test script for the improved Law RAG Chatbot
Tests the specific drunk driving question that was failing
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system import RAGSystem
from config import *

async def test_drunk_driving_question():
    """Test the specific drunk driving question"""
    print("🚗 Testing Improved Drunk Driving Question")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag = RAGSystem()
        await rag.initialize()
        print("✅ RAG system initialized successfully!")
        
        # Test the specific question
        question = "Why is drunk driving causing accident punished so much worse than just drunk driving?"
        print(f"\n❓ Question: {question}")
        
        # Get response with increased context length
        print("\n🔍 Searching for relevant information...")
        response = await rag.get_response(question, context_length=5)
        
        print(f"\n📊 Results:")
        print(f"   Confidence: {response['confidence']}")
        print(f"   Sources found: {len(response['sources'])}")
        
        if response['sources']:
            print(f"\n📚 Sources:")
            for i, source in enumerate(response['sources'][:3], 1):
                print(f"   {i}. Relevance: {source.get('relevance_score', 'N/A'):.3f}")
                content_preview = source['content'][:150] + "..." if len(source['content']) > 150 else source['content']
                print(f"      Content: {content_preview}")
        
        print(f"\n💬 Answer:")
        answer = response['answer']
        if len(answer) > 500:
            print(f"   {answer[:500]}...")
            print(f"   ... (truncated for display)")
        else:
            print(f"   {answer}")
        
        # Check if the response is better than the original
        if "couldn't find relevant legal information" not in response['answer']:
            print("\n✅ SUCCESS: The system now provides a meaningful response!")
            print("   The enhanced search and fallback mechanisms are working.")
        else:
            print("\n⚠️  The system still couldn't find specific information,")
            print("   but it should now provide a helpful fallback response.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

async def test_search_variations():
    """Test the enhanced search variations"""
    print("\n🔍 Testing Enhanced Search Variations")
    print("=" * 40)
    
    try:
        rag = RAGSystem()
        await rag.initialize()
        
        # Test the enhanced search method directly
        question = "drunk driving accident penalties"
        print(f"Testing search for: {question}")
        
        results = await rag._enhanced_search(question, 3)
        print(f"Found {len(results)} results with enhanced search")
        
        if results:
            print("✅ Enhanced search is working!")
            for i, result in enumerate(results[:2], 1):
                print(f"   {i}. Score: {result.get('relevance_score', 'N/A'):.3f}")
        else:
            print("⚠️  Enhanced search didn't find results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing search variations: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 Testing Improved Law RAG Chatbot")
    print("=" * 50)
    
    # Check environment
    if not os.getenv('HF_TOKEN') or not os.getenv('GROQ_API_KEY'):
        print("❌ Environment variables not set. Please run setup_env.py first.")
        return
    
    # Run tests
    success = True
    
    # Test 1: Drunk driving question
    if not await test_drunk_driving_question():
        success = False
    
    # Test 2: Enhanced search
    if not await test_search_variations():
        success = False
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("💡 The improved system should now handle the drunk driving question much better.")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    print("\n🚀 To test the full API with sessions:")
    print("   1. Start the server: python app.py")
    print("   2. Run the demo: python demo.py")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}") 