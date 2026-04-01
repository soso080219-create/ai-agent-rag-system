"""
전체 파이프라인 통합
"""

from typing import Dict
from .intent_analyzer import IntentAnalyzer
from .rag_searcher import RAGSearcher
from .plos_searcher import PLoSSearcher
from .persona_engine import PersonaEngine

class SmartRAGPipeline:
    def __init__(
        self,
        rag_db_path: str,
        use_claude: bool = False,
        claude_api_key: str = None
    ):
        # 각 모듈 초기화
        self.intent_analyzer = IntentAnalyzer(claude_api_key)
        self.rag_searcher = RAGSearcher(rag_db_path)
        self.plos_searcher = PLoSSearcher()
        self.persona_engine = PersonaEngine(use_claude, claude_api_key)
    
    def process(self, question: str, persona_key: str = "basic") -> Dict:
        """
        전체 파이프라인 실행
        """
        print(f"\n{'='*70}")
        print(f"📝 Question: {question}")
        print(f"🎭 Persona: {persona_key}")
        print(f"{'='*70}\n")
        
        # [1] 의도 파악
        print("📊 Step 1: Analyzing intent...")
        intent = self.intent_analyzer.analyze(question)
        print(f"   Category: {intent['category']}")
        print(f"   Needs evidence: {intent['needs_evidence']}")
        print()
        
        # [2] Context 검색 (3단계 Fallback)
        print("🔍 Step 2: Searching for context...")
        context, source_type = self._search_with_fallback(question, intent)
        print(f"   ✅ Source: {source_type}")
        print(f"   Context length: {len(context) if context else 0} chars")
        print()
        
        # [3] 답변 생성
        print("💬 Step 3: Generating response...")
        response = self.persona_engine.generate(
            question=question,
            context=context or "",
            source_type=source_type,
            persona_key=persona_key
        )
        
        print(f"\n{'='*70}")
        print(f"✨ Response:\n{response}")
        print(f"{'='*70}\n")
        
        return {
            "question": question,
            "intent": intent,
            "source_type": source_type,
            "context_preview": (context[:200] + "...") if context else None,
            "persona": persona_key,
            "answer": response
        }
    
    def _search_with_fallback(self, question: str, intent: dict) -> tuple:
        """
        3단계 Fallback 검색
        
        Returns:
            (context, source_type)
        """
        # [1단계] Local RAG 시도
        context, found = self.rag_searcher.search(question, intent)
        if found:
            return context, "local_rag"
        
        # [2단계] PLoS API 시도
        context, found = self.plos_searcher.search(question, intent)
        if found:
            return context, "plos_api"
        
        # [3단계] Claude 자체 지식
        print("ℹ️  Using LLM's own knowledge (no external context)")
        return None, "claude_knowledge"