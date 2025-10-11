"""
Enhanced Query Mixin for RAGAnything

This module provides enhanced query functionality with EFR layer integration.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
import logging
from lightrag import QueryParam


class EnhancedQueryMixin:
    """Enhanced query functionality with EFR layer support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def aquery_enhanced(self, query: str, mode: str = "mix", use_efr: bool = True, **kwargs) -> str:
        """
        Enhanced query with EFR layer support
        
        Args:
            query: User query
            mode: LightRAG query mode
            use_efr: Whether to use EFR layer
            **kwargs: Additional query parameters
            
        Returns:
            Enhanced query result
        """
        self.logger.info(f"Executing enhanced query: {query[:100]}...")
        
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()
        
        if use_efr and hasattr(self, 'efr_layer') and self.efr_layer:
            # Use EFR layer for enhanced retrieval
            return await self._query_with_efr(query, mode, **kwargs)
        else:
            # Fallback to standard query
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(query, param=query_param)
    
    async def _query_with_efr(self, query: str, mode: str, **kwargs) -> str:
        """Query with EFR layer processing"""
        try:
            self.logger.info(f"🔧 EFR Layer Processing: {query[:50]}...")

            # First, get the standard LightRAG response to ensure we have retrieval results
            query_param = QueryParam(mode=mode, **kwargs)
            standard_response = await self.lightrag.aquery(query, param=query_param)
            
            # Now get the raw prompt for EFR processing
            query_param_prompt = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
            raw_prompt = await self.lightrag.aquery(query, param=query_param_prompt)

            # Extract retrieval results from the prompt
            retrieval_results = self._extract_retrieval_results_from_prompt(raw_prompt)
            self.logger.info(f"🔧 Extracted {len(retrieval_results)} retrieval results")

            # If we have retrieval results, process with EFR layer
            if retrieval_results and len(retrieval_results) > 0:
                if hasattr(self.efr_layer, 'process_retrieval_results'):
                    self.logger.info("🔧 Processing with EFR layer...")
                    if asyncio.iscoroutinefunction(self.efr_layer.process_retrieval_results):
                        processed_results = await self.efr_layer.process_retrieval_results(
                            retrieval_results, query, self.llm_model_func
                        )
                    else:
                        processed_results = self.efr_layer.process_retrieval_results(
                            retrieval_results, query
                        )
                    self.logger.info(f"🔧 EFR processed {len(processed_results)} results")
                    
                    # Generate enhanced response using processed results
                    enhanced_prompt = self._build_enhanced_prompt(processed_results, query)
                    self.logger.info(f"🔧 Built enhanced prompt with {len(enhanced_prompt)} characters")

                    # Use LLM directly with enhanced prompt
                    if hasattr(self, 'llm_model_func') and self.llm_model_func:
                        self.logger.info("🔧 Generating response with LLM using EFR-enhanced prompt")
                        response = await self.llm_model_func(enhanced_prompt)
                        self.logger.info(f"🔧 Generated response: {len(response)} characters")
                    else:
                        self.logger.warning("🔧 No LLM function available, using standard response")
                        response = standard_response
                else:
                    self.logger.warning("🔧 EFR layer has no process_retrieval_results method")
                    response = standard_response
            else:
                self.logger.warning("🔧 No retrieval results found, using standard response")
                response = standard_response

            self.logger.info("✅ Enhanced query with EFR completed")
            return response

        except Exception as e:
            self.logger.warning(f"❌ EFR processing failed: {e}, falling back to standard query")
            import traceback
            traceback.print_exc()
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(query, param=query_param)
    
    def _extract_retrieval_results_from_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """Extract retrieval results from prompt"""
        results = []
        
        # Check if prompt contains actual content (not just error messages)
        if not prompt or "Sorry, I'm not able to provide" in prompt or len(prompt.strip()) < 100:
            self.logger.warning("🔧 Prompt contains insufficient content for EFR processing")
            return results
        
        # Parse the prompt to extract meaningful sections
        # Look for different types of content sections
        sections = []
        
        # Split by double newlines first
        raw_sections = prompt.split('\n\n')
        
        for section in raw_sections:
            section = section.strip()
            if len(section) > 100:  # Only include substantial sections
                sections.append(section)
        
        # If we have good sections, use them
        if sections:
            for i, section in enumerate(sections):
                # Calculate relevance score based on position and content
                base_score = 0.9 - (i * 0.1)
                
                # Boost score for sections with key indicators
                if any(keyword in section.lower() for keyword in [
                    "contribution", "technology", "framework", "experiment", "result", "limitation"
                ]):
                    base_score += 0.1
                
                results.append({
                    "content": section,
                    "score": min(1.0, base_score),
                    "metadata": {"source": "document", "section": i, "type": "text_chunk"},
                    "modality": "text",
                    "language": "en",
                    "source": "document",
                    "reliability_score": 0.9
                })
        
        # If no good sections found, try to extract from the full prompt
        if not results and len(prompt) > 200:
            # Split into smaller chunks
            chunk_size = 500
            chunks = [prompt[i:i+chunk_size] for i in range(0, len(prompt), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 100:
                    results.append({
                        "content": chunk.strip(),
                        "score": 0.8 - (i * 0.1),
                        "metadata": {"source": "document", "chunk": i, "type": "full_text"},
                        "modality": "text",
                        "language": "en",
                        "source": "document",
                        "reliability_score": 0.8
                    })
        
        self.logger.info(f"🔧 Extracted {len(results)} meaningful sections from prompt")
        return results
    
    def _build_enhanced_prompt(self, results: List[Any], query: str) -> str:
        """Build enhanced prompt from processed results"""
        if not results:
            return f"Query: {query}\n\nNo relevant information found."
        
        enhanced_prompt = f"Based on the following information from the document, please provide a detailed answer to: {query}\n\n"
        enhanced_prompt += "Document Context:\n"
        
        for i, result in enumerate(results, 1):
            if hasattr(result, 'content'):
                content = result.content
            else:
                content = str(result)
            
            # Include more content, not truncated
            enhanced_prompt += f"{i}. {content}\n\n"
        
        enhanced_prompt += "Please provide a comprehensive and detailed answer based on the above document context. If the information is insufficient for a complete answer, provide the best possible response based on the available information."
        
        return enhanced_prompt
    
    async def _ensure_lightrag_initialized(self):
        """Ensure LightRAG is initialized"""
        if not hasattr(self, 'lightrag') or self.lightrag is None:
            raise ValueError("LightRAG not initialized. Please initialize RAGAnything first.")
        
        # Call parent class method if available
        if hasattr(super(), '_ensure_lightrag_initialized'):
            await super()._ensure_lightrag_initialized()
