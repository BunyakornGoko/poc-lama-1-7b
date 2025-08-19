#!/usr/bin/env python3
"""
Test script for AdaptLLM/finance-LLM
Verifies the model works correctly with basic functionality
"""

import sys
import time
from finance_llm_demo import FinanceLLMDemo

def test_basic_functionality():
    """Test basic model functionality"""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        # Initialize model
        print("1. Initializing model...")
        demo = FinanceLLMDemo()
        print("✅ Model initialization successful")
        
        # Test basic question
        print("\n2. Testing basic finance question...")
        test_question = "What is a stock?"
        response = demo.generate_response(test_question, max_length=100)
        
        if response and len(response) > 10:
            print("✅ Basic question test successful")
            print(f"   Question: {test_question}")
            print(f"   Response: {response[:100]}...")
        else:
            print("❌ Basic question test failed")
            return False
        
        # Test financial document analysis
        print("\n3. Testing financial document analysis...")
        doc_question = """Use this fact to answer the question:
        Company: Apple Inc.
        Stock Symbol: AAPL
        Exchange: NASDAQ
        Sector: Technology
        
        What is Apple's stock symbol?"""
        
        doc_response = demo.generate_response(doc_question, max_length=150)
        
        if doc_response and len(doc_response) > 10:
            print("✅ Document analysis test successful")
            print(f"   Response: {doc_response[:100]}...")
        else:
            print("❌ Document analysis test failed")
            return False
        
        print("\n🎉 All tests passed! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_performance():
    """Test model performance"""
    print("\n⚡ Testing Performance")
    print("=" * 40)
    
    try:
        demo = FinanceLLMDemo()
        
        # Test response time
        test_questions = [
            "What is a bond?",
            "Explain inflation",
            "What is a mutual fund?"
        ]
        
        total_time = 0
        successful_responses = 0
        
        for i, question in enumerate(test_questions, 1):
            print(f"   Testing question {i}/{len(test_questions)}...")
            
            start_time = time.time()
            response = demo.generate_response(question, max_length=100)
            end_time = time.time()
            
            response_time = end_time - start_time
            total_time += response_time
            
            if response and len(response) > 10:
                successful_responses += 1
                print(f"   ✅ Response time: {response_time:.2f}s")
            else:
                print(f"   ❌ Failed to get response")
        
        avg_time = total_time / len(test_questions)
        success_rate = (successful_responses / len(test_questions)) * 100
        
        print(f"\n📊 Performance Results:")
        print(f"   Average response time: {avg_time:.2f}s")
        print(f"   Success rate: {success_rate:.1f}%")
        
        return success_rate >= 80  # 80% success rate threshold
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏦 AdaptLLM Finance-LLM Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    basic_test_passed = test_basic_functionality()
    
    if not basic_test_passed:
        print("\n❌ Basic functionality test failed. Stopping tests.")
        sys.exit(1)
    
    # Test performance
    performance_test_passed = test_performance()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    print(f"Basic Functionality: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"Performance: {'✅ PASSED' if performance_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and performance_test_passed:
        print("\n🎉 All tests passed! Your model is ready to use.")
        print("💡 You can now run the full demo with: python finance_llm_demo.py")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
