#!/usr/bin/env python3
"""
Comprehensive SPIQA Test B with detailed success rate analysis
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import base64
from difflib import SequenceMatcher

import asyncio
import urllib.request
import urllib.error
import json
import os

def _http_post_json(url: str, payload: dict, timeout: int = 240) -> dict:
    """Minimal helper to POST JSON without external deps."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} error from {url}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e}")

def build_ollama_llm_func():
    """Create an LLM function that talks to Ollama via HTTP."""
    host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct").strip()
    url = f"{host}/api/generate"
    timeout = 240

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                full_prompt += f"System: {content}\n\n"
            elif role == "user":
                full_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                full_prompt += f"Assistant: {content}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant:"

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False
        }

        try:
            resp = await asyncio.to_thread(_http_post_json, url, payload, timeout)
            return resp.get("response", "")
        except Exception as e:
            logger.error(f"Ollama LLM request failed: {e}")
            return ""

    return llm_func

def build_ollama_vision_func():
    """Create a vision function that talks to Ollama via HTTP."""
    host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("VISION_MODEL", "llava:7b").strip()
    url = f"{host}/api/generate"
    timeout = 240

    async def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        if messages:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
        elif image_data:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            })
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history_messages:
                messages.append(msg)
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }

        try:
            resp = await asyncio.to_thread(_http_post_json, url, payload, timeout)
            return resp.get("response", "")
        except Exception as e:
            logger.error(f"Ollama Vision request failed: {e}")
            return ""

    return vision_func

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent

class ComprehensiveSPIQATester:
    def __init__(self, image_root: str):
        self.image_root = image_root
        self.llm_func = build_ollama_llm_func()
        self.vision_func = build_ollama_vision_func()
        
    def similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for comparison"""
        # Simple key phrase extraction
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        key_phrases = [word for word in words if word not in stop_words and len(word) > 2]
        return key_phrases
    
    def evaluate_answer(self, predicted: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate the predicted answer against ground truth"""
        # Basic similarity
        similarity_score = self.similarity(predicted, ground_truth)
        
        # Key phrase overlap
        pred_phrases = set(self.extract_key_phrases(predicted))
        gt_phrases = set(self.extract_key_phrases(ground_truth))
        
        if len(gt_phrases) > 0:
            phrase_overlap = len(pred_phrases.intersection(gt_phrases)) / len(gt_phrases)
        else:
            phrase_overlap = 0.0
        
        # Check if ground truth is contained in prediction
        contains_gt = ground_truth.lower() in predicted.lower()
        
        # Check if prediction is contained in ground truth
        contains_pred = predicted.lower() in ground_truth.lower()
        
        # Overall correctness (using multiple criteria)
        is_correct = (
            similarity_score > 0.7 or  # High similarity
            phrase_overlap > 0.5 or    # Good phrase overlap
            contains_gt or             # Contains ground truth
            contains_pred               # Prediction is contained in ground truth
        )
        
        return {
            "similarity_score": similarity_score,
            "phrase_overlap": phrase_overlap,
            "contains_gt": contains_gt,
            "contains_pred": contains_pred,
            "is_correct": is_correct,
            "pred_phrases": list(pred_phrases),
            "gt_phrases": list(gt_phrases)
        }
    
    async def process_paper(self, paper_id: str, paper_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        
        # Extract text content from passages if available
        text_content = " ".join(paper_content.get("passages", []))
        
        # Extract figure captions and content
        figures_data = paper_content.get("all_figures_tables", {})
        figure_texts = []
        for fig_id, fig_info in figures_data.items():
            if isinstance(fig_info, dict):
                caption = fig_info.get("caption", "")
                content_type = fig_info.get("content_type", "figure")
                figure_texts.append(f"{content_type.capitalize()} {fig_id}: {caption}")
            else:
                # If fig_info is a string, use it directly
                figure_texts.append(f"Figure {fig_id}: {fig_info}")
        
        # Combine all textual information for context
        full_context_text = text_content + "\n\n" + "\n".join(figure_texts)

        questions = paper_content.get("question", [])
        answers = paper_content.get("answer", [])
        question_types = paper_content.get("question_type", [])
        evidential_infos = paper_content.get("evidential_info", [])

        print(f"  📝 Questions: {len(questions)}")
        print(f"  🖼️  Figures: {len(figures_data)}")

        for q_idx, question_text in enumerate(questions):
            pid = f"{paper_id}_q{q_idx}"
            gt_answer_text = answers[q_idx] if q_idx < len(answers) else ""
            question_type = question_types[q_idx] if q_idx < len(question_types) else ""
            evidential_info = evidential_infos[q_idx] if q_idx < len(evidential_infos) else []

            print(f"\n  ❓ Question {q_idx + 1}: {question_text}")
            
            # Find relevant figures based on evidential_info
            relevant_figures = []
            for evidence_piece in evidential_info:
                reference = evidence_piece.get("reference")
                if reference and reference in figures_data:
                    relevant_figures.append(reference)
            
            # For simplicity, we'll just use the first evidential info's context as direct context for the LLM
            context_for_llm = full_context_text
            if evidential_info:
                # Prioritize context from evidential_info if available and specific
                first_evidence_context = evidential_info[0].get("context")
                if first_evidence_context:
                    context_for_llm = first_evidence_context
            
            print(f"    📊 Evidence pieces: {len(evidential_info)}")
            print(f"    🖼️  Relevant figures: {len(relevant_figures)}")

            try:
                # Construct a prompt for Ollama
                prompt = f"Given the following context:\n{context_for_llm}\n\nQuestion: {question_text}\nAnswer:"
                
                # If there are relevant images, try to include them
                image_b64 = None
                if relevant_figures:
                    # For simplicity, take the first relevant image
                    first_figure_ref = relevant_figures[0]
                    # Assuming image files are named after their references and are in the image_root
                    image_path = Path(self.image_root) / first_figure_ref
                    if image_path.exists():
                        # Read image and convert to base64
                        with open(image_path, "rb") as img_file:
                            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                        logger.info(f"Attached image: {image_path}")
                    else:
                        logger.warning(f"Image not found: {image_path}")

                # Use Ollama directly
                if image_b64:
                    answer_text = await self.vision_func(prompt=prompt, image_data=image_b64, system_prompt="You are a helpful multimodal assistant. Provide a concise answer.")
                else:
                    answer_text = await self.llm_func(prompt=prompt, system_prompt="You are a helpful assistant. Provide a concise answer.")

                # Evaluate the answer
                evaluation = self.evaluate_answer(answer_text, gt_answer_text)
                
                print(f"    🤖 Answer: {answer_text[:100]}...")
                print(f"    📊 Similarity: {evaluation['similarity_score']:.3f}")
                print(f"    📈 Phrase Overlap: {evaluation['phrase_overlap']:.3f}")
                print(f"    ✅ Correct: {evaluation['is_correct']}")
                
                results.append({
                    "question": question_text,
                    "context_length": len(context_for_llm),
                    "evidence_count": len(evidential_info),
                    "figures_count": len(figures_data),
                    "answer": answer_text,
                    "ground_truth": gt_answer_text,
                    "relevant_figures": relevant_figures,
                    "question_type": question_type,
                    "paper_id": paper_id,
                    "question_index": q_idx,
                    "evaluation": evaluation
                })
            except Exception as e:
                logger.error(f"Error processing question {pid}: {e}")
                print(f"    ❌ Error processing question: {e}")
                results.append({
                    "question": question_text,
                    "error": str(e),
                    "paper_id": paper_id,
                    "question_index": q_idx
                })
        return results

async def main(args):
    # Ensure the dataset is downloaded and extracted
    spiqa_data_path = Path(BASE_DIR) / "dataset" / "test-B" / "SPIQA_testB.json"
    spiqa_image_root = Path(BASE_DIR) / "dataset" / "test-B" / "SPIQA_testB_Images"

    print(f"Looking for SPIQA data at: {spiqa_data_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"BASE_DIR: {BASE_DIR}")
    
    if not spiqa_data_path.exists():
        print(f"Error: SPIQA_testB.json not found at {spiqa_data_path}. Please run download_spiqa.py first.")
        return
    
    if not spiqa_image_root.exists():
        print(f"Warning: SPIQA_testB_Images directory not found at {spiqa_image_root}. Proceeding without images.")
        spiqa_image_root = None

    # Load SPIQA test B data
    with open(spiqa_data_path, "r", encoding="utf-8") as f:
        spiqa_raw_data = json.load(f)

    print("🚀 Starting Comprehensive SPIQA Test B processing...")
    print(f"📁 Data path: {spiqa_data_path}")
    print(f"🖼️  Image root: {spiqa_image_root}")
    print(f"📊 Loaded {len(spiqa_raw_data)} papers from SPIQA test B")

    tester = ComprehensiveSPIQATester(str(spiqa_image_root) if spiqa_image_root else "")

    all_results = {}
    processed_count = 0
    failed_count = 0
    correct_count = 0
    
    # Statistics
    similarity_scores = []
    phrase_overlaps = []
    question_types = {}
    paper_stats = {}

    # Limit to a few papers for initial testing
    papers_to_process = list(spiqa_raw_data.items())
    if args.test_number > 0:
        papers_to_process = papers_to_process[:args.test_number]

    for paper_id, paper_content in papers_to_process:
        print(f"\n🔬 Processing paper: {paper_id}")
        paper_results = await tester.process_paper(paper_id, paper_content)
        
        paper_correct = 0
        paper_total = 0
        
        for res in paper_results:
            qid = f"{res['paper_id']}_q{res['question_index']}"
            all_results[qid] = res
            
            if "error" in res:
                failed_count += 1
            else:
                processed_count += 1
                paper_total += 1
                
                # Collect statistics
                if "evaluation" in res:
                    evaluation = res["evaluation"]
                    similarity_scores.append(evaluation["similarity_score"])
                    phrase_overlaps.append(evaluation["phrase_overlap"])
                    
                    if evaluation["is_correct"]:
                        correct_count += 1
                        paper_correct += 1
                    
                    # Question type statistics
                    q_type = res.get("question_type", "Unknown")
                    if q_type not in question_types:
                        question_types[q_type] = {"total": 0, "correct": 0}
                    question_types[q_type]["total"] += 1
                    if evaluation["is_correct"]:
                        question_types[q_type]["correct"] += 1
        
        paper_stats[paper_id] = {
            "total_questions": paper_total,
            "correct_questions": paper_correct,
            "accuracy": paper_correct / paper_total if paper_total > 0 else 0
        }

    # Calculate overall statistics
    overall_accuracy = correct_count / processed_count if processed_count > 0 else 0
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    avg_phrase_overlap = sum(phrase_overlaps) / len(phrase_overlaps) if phrase_overlaps else 0

    print("\n" + "="*60)
    print("📈 COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    print(f"📊 Total questions processed: {processed_count + failed_count}")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"🎯 Correct answers: {correct_count}")
    print(f"📈 Overall Accuracy: {overall_accuracy:.3f} ({correct_count}/{processed_count})")
    print(f"📊 Average Similarity Score: {avg_similarity:.3f}")
    print(f"📊 Average Phrase Overlap: {avg_phrase_overlap:.3f}")
    
    print(f"\n📋 Question Type Performance:")
    for q_type, stats in question_types.items():
        type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {q_type}: {type_accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    print(f"\n📄 Paper-wise Performance:")
    for paper_id, stats in paper_stats.items():
        print(f"  {paper_id}: {stats['accuracy']:.3f} ({stats['correct_questions']}/{stats['total_questions']})")

    print("\n📋 Sample Results:")
    for i, (qid, res) in enumerate(all_results.items()):
        if i >= 5:  # Print first 5 samples
            break
        if "evaluation" in res:
            eval_info = res["evaluation"]
            print(f"  {qid}: {res.get('answer', '')[:80]}...")
            print(f"    Similarity: {eval_info['similarity_score']:.3f}, Correct: {eval_info['is_correct']}")

    output_file = "spiqa_comprehensive_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Comprehensive SPIQA Test B with detailed analysis.")
    parser.add_argument("--test_number", type=int, default=5,
                        help="Number of papers to process for testing. Set to 0 for all.")
    args = parser.parse_args()
    asyncio.run(main(args))
