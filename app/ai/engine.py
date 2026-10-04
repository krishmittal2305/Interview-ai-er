"""
Assessment engine — orchestrates structured AI prompts with telemetry and retry safety.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.ai.providers.base import AIProvider
from app.ai.schemas import (
    AnswerEvaluation,
    CodeEvaluation,
    Question,
    SessionSynthesis,
    SkillExtraction,
    AIRun,
)

logger = logging.getLogger(__name__)


class AssessmentEngine:
    def __init__(self, provider: AIProvider):
        self.provider = provider
        self.interview_types = {
            "technical": "Technical programming and computer science questions",
            "behavioral": "Soft skills and behavioral questions",
            "system_design": "System architecture and design questions",
            "coding": "Practical coding problems and algorithms",
            "Software Engineer": "Software engineering fundamentals and architecture",
            "Data Scientist": "Machine learning, statistical analysis, and data modeling",
            "Product Manager": "Product strategy, user empathy, and technical tradeoffs",
            "DevOps Engineer": "CI/CD, container orchestration, and reliability engineering",
        }

    def _execute_with_telemetry(self, prompt: str, schema: type, task_type: str = "unknown") -> Dict[str, Any]:
        """Executes a generation task and wraps it with latency and metadata."""
        start_time = time.time()
        telemetry_data = {}

        try:
            result, telemetry_data = self.provider.generate_structured(prompt, schema)
            status = "success"
            error = None
            data = result.model_dump()
        except Exception as e:
            status = "failed"
            error = str(e)
            data = None

        latency = time.time() - start_time
        metadata = self.provider.get_metadata()
        model_name = metadata.get("model_name") or metadata.get("provider", "llm-provider")
        
        # Build AIRun model instance for legacy telemetry
        ai_run = AIRun(
            task_type=task_type,
            model=metadata.get("provider", "unknown"),
            model_version=model_name,
            latency=latency,
            input_tokens=telemetry_data.get("input_tokens"),
            output_tokens=telemetry_data.get("output_tokens"),
            schema_validation=telemetry_data.get("schema_validation", False) if error is None else False,
            retry_count=telemetry_data.get("retry_count", 0),
            failure_state=error,
            confidence=data.get("confidence") if data else None,
            evaluation_metadata={"status": status},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        logger.info(f"AIRun Telemetry: {ai_run.model_dump_json()}")
        metadata.update(ai_run.model_dump())

        # Production Unified Observability Tracker for LLM
        try:
            from app.observability.tracker import get_telemetry_tracker
            tracker = get_telemetry_tracker()
            tracker.record_inference(
                inference_type="llm",
                model=model_name,
                model_version=metadata.get("provider", "v1.0"),
                task=task_type,
                latency_ms=round(latency * 1000.0, 2),
                success=(error is None),
                status="SUCCESS" if error is None else "FAILED",
                error_message=error,
                input_size=len(prompt),
                output_size=len(str(data)) if data else 0,
                input_tokens=telemetry_data.get("input_tokens"),
                output_tokens=telemetry_data.get("output_tokens"),
                memory_device="api/cloud",
                fallback_usage=telemetry_data.get("fallback_usage", False),
                retry_count=telemetry_data.get("retry_count", 0),
                metadata={"confidence": data.get("confidence") if data else None},
            )
        except Exception as tel_err:
            logger.debug(f"Failed to record LLM observability span: {tel_err}")

        if error:
            raise RuntimeError(f"AI Generation Failed: {error}")

        return {
            "data": data,
            "metadata": metadata,
        }

    def generate_question(
        self,
        interview_type: str,
        difficulty: str = "intermediate",
        topic: Optional[str] = None,
        skill_focus: Optional[str] = None,
        target_role: Optional[str] = None,
        excluded_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate an interview question using the constrained schema with duplicate prevention."""
        context_parts = []
        if target_role:
            context_parts.append(f"Target Role: {target_role}")
        if skill_focus:
            context_parts.append(f"Core Skill Focus: {skill_focus}")
        if topic:
            context_parts.append(f"Topic: {topic}")
        context_parts.append(f"Interview Type: {self.interview_types.get(interview_type, interview_type)}")

        context = "\n".join(context_parts)

        exclusions_text = ""
        if excluded_questions:
            exclusions_list = "\n".join([f"- {q}" for q in excluded_questions[-10:]])
            exclusions_text = f"\nCRITICAL: Avoid asking any of the following questions or variations:\n{exclusions_list}\n"

        prompt = f"""
        Generate a {difficulty} difficulty interview question.

        Requirements:
        - Question should be clear, specific, and realistic for a real tech interview
        - Tailor the question specifically to the Skill Focus and Target Role
        - Include context and constraints if applicable
        - Difficulty should strictly match {difficulty} level
        - Describe expected reasoning and rubric clearly
        {exclusions_text}
        Context:
        {context}
        """

        response = self._execute_with_telemetry(prompt, Question, task_type="generate_question")
        return {**response["data"], "_metadata": response["metadata"]}

    def evaluate_answer(self, question: str, answer: str, question_type: str = "technical",
                        evaluation_rubric: str = '', expected_concepts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a text answer using the hybrid ML + Gemini pipeline.

        Steps:
            1. Generate ML-derived concept coverage evidence (DeBERTa NLI + MiniLM)
            2. Inject structured evidence into the Gemini evaluation prompt
            3. Gemini reasons over the evidence to produce the final evaluation
        """
        # Step 1: Generate ML evidence
        ml_evidence = None
        ml_evidence_block = ""
        try:
            from ml.pipelines.hybrid_assessment import get_hybrid_pipeline
            pipeline = get_hybrid_pipeline()
            ml_evidence = pipeline.generate_ml_evidence(
                candidate_answer=answer,
                question_text=question,
                evaluation_rubric=evaluation_rubric or question,
                expected_concepts=expected_concepts,
            )
            ml_evidence_block = pipeline.format_evidence_for_gemini(ml_evidence)
            logger.info(
                "ML evidence generated: coverage=%.1f%%, concepts=%d, time=%.1fms",
                ml_evidence.get("ml_concept_coverage", {}).get("overall_coverage_pct", 0),
                ml_evidence.get("concepts_evaluated", 0),
                ml_evidence.get("total_pipeline_time_ms", 0),
            )
        except Exception as e:
            logger.warning("Hybrid ML evidence generation failed (non-fatal): %s", e)

        # Step 2: Build the evaluation prompt with ML evidence
        base_prompt = f"""
        Evaluate this interview answer for a {question_type} question.

        Question: {question}
        Answer: {answer}

        You are an expert technical interviewer. Score the answer out of 100 on multiple axes.
        Provide constructive feedback, strengths, and weaknesses. Be honest and critical.
        """

        if ml_evidence_block:
            prompt = f"{base_prompt}\n{ml_evidence_block}"
        else:
            prompt = base_prompt

        # Step 3: Gemini evaluates with ML evidence context
        response = self._execute_with_telemetry(prompt, AnswerEvaluation, task_type="evaluate_answer")
        result = {**response["data"], "_metadata": response["metadata"]}

        # Attach the raw ML evidence and model attribution to the response
        cov = ml_evidence.get("ml_concept_coverage", {}) if ml_evidence else {}
        result["ml_concept_coverage"] = cov
        result["ml_pipeline_version"] = ml_evidence.get("pipeline_version") if ml_evidence else "hybrid_v1.0"
        result["ml_signal_type"] = cov.get("signal_type") if cov else None

        # Mandatory model attribution fields
        result["model_name"] = cov.get("model_name", "answer-nli-v1")
        result["model_version"] = cov.get("model_version", "1.0.0")
        result["dataset_version"] = cov.get("dataset_version", "mnli-snli-v1")
        result["training_run"] = cov.get("training_run", "answer-nli-v1_pretrained_deberta_minilm")
        result["inference_timestamp"] = cov.get("inference_timestamp", datetime.now(timezone.utc).isoformat())
        result["model_attribution"] = {
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "dataset_version": result["dataset_version"],
            "training_run": result["training_run"],
            "inference_timestamp": result["inference_timestamp"],
        }

        return result

    def evaluate_code(self, code: str, language: str, question: str) -> Dict[str, Any]:
        """
        Evaluate a code submission using the hybrid ML + Gemini pipeline.

        Steps:
            1. Run CodeBERT defect detection for ML-derived code-risk signal
            2. Inject structured evidence into the Gemini evaluation prompt
            3. Gemini reasons over the evidence to produce the final evaluation
        """
        # Step 1: Generate ML defect detection evidence
        ml_defect = None
        ml_evidence_block = ""
        try:
            from ml.models.defect_detector import get_defect_detector
            detector = get_defect_detector()
            ml_defect = detector.analyze_code(code, language=language)
            logger.info(
                "ML defect detection: prob=%.3f, risk=%s, method=%s, time=%.1fms",
                ml_defect.get("defect_probability", 0),
                ml_defect.get("risk_band", "unknown"),
                ml_defect.get("method", "unknown"),
                ml_defect.get("inference_time_ms", 0),
            )

            risk_indicators = ml_defect.get("risk_indicators", [])
            indicators_text = "\n".join(f"  - {r}" for r in risk_indicators) if risk_indicators else "  (none detected)"

            ml_evidence_block = f"""
═══════════════════════════════════════════════════════════════
ML-DERIVED CODE-RISK SIGNAL (Structured Evidence)
═══════════════════════════════════════════════════════════════
Model: {ml_defect.get('model_version', 'unknown')}
Method: {ml_defect.get('method', 'unknown')}
Defect Probability: {ml_defect.get('defect_probability', 0):.3f}
Risk Band: {ml_defect.get('risk_band', 'unknown').upper()}
Confidence: {ml_defect.get('confidence', 0):.3f}

Static Risk Indicators:
{indicators_text}

IMPORTANT INSTRUCTIONS:
  1. This is an ML-derived code-risk signal, NOT proof of incorrectness.
  2. Use this signal to guide your analysis — inspect areas of higher risk more carefully.
  3. Executable tests remain authoritative for runtime correctness.
═══════════════════════════════════════════════════════════════
"""
        except Exception as e:
            logger.warning("ML defect detection failed (non-fatal): %s", e)

        # Step 2: Build the evaluation prompt with ML evidence
        base_prompt = f"""
        Evaluate this {language} code submission for the following question:

        Question: {question}
        Code:
        ```{language}
        {code}
        ```

        Analyze correctness, time/space complexity, and code quality.
        Identify strengths and issues. Provide actionable recommendations.
        """

        if ml_evidence_block:
            prompt = f"{base_prompt}\n{ml_evidence_block}"
        else:
            prompt = base_prompt

        # Step 3: Gemini evaluates with ML evidence context
        response = self._execute_with_telemetry(prompt, CodeEvaluation, task_type="evaluate_code")
        result = {**response["data"], "_metadata": response["metadata"]}

        # Attach the raw ML evidence and model attribution to the response
        defect_data = ml_defect or {}
        result["ml_defect_detection"] = {
            "source": defect_data.get("source", "ml"),
            "defect_probability": defect_data.get("defect_probability", 0.0),
            "risk_band": defect_data.get("risk_band", "unknown"),
            "model_version": defect_data.get("model_version", "1.0.0"),
            "confidence": defect_data.get("confidence", 0.0),
            "inference_time_ms": defect_data.get("inference_time_ms", 0),
            "method": defect_data.get("method", "unknown"),
            "risk_indicators": defect_data.get("risk_indicators", []),
            "is_fine_tuned": defect_data.get("is_fine_tuned", False),
        }

        # Mandatory model attribution fields
        result["model_name"] = defect_data.get("model_name", "code-risk-v1")
        result["model_version"] = defect_data.get("model_version", "1.0.0")
        result["dataset_version"] = defect_data.get("dataset_version", "CodeXGLUE-defect-v1")
        result["training_run"] = defect_data.get("training_run", "code-risk-v1_codebert_ast_analysis")
        result["inference_timestamp"] = defect_data.get("inference_timestamp", datetime.now(timezone.utc).isoformat())
        result["model_attribution"] = {
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "dataset_version": result["dataset_version"],
            "training_run": result["training_run"],
            "inference_timestamp": result["inference_timestamp"],
        }

        return result

    def generate_follow_up_question(
        self,
        original_question: str,
        candidate_answer: str,
        evaluation: Dict[str, Any],
        weakness: Optional[str] = None,
        skill_focus: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a targeted follow-up question based on previous answer and specific weakness."""
        targeted_clause = ""
        if weakness:
            targeted_clause += f"\nSpecific weakness to probe: {weakness}"
        if skill_focus:
            targeted_clause += f"\nSkill focus: {skill_focus}"
        if difficulty:
            targeted_clause += f"\nTarget difficulty: {difficulty}"

        prompt = f"""
        Based on this interview exchange, generate a targeted follow-up question.

        Original Question: {original_question}
        Candidate Answer: {candidate_answer}
        Identified Weaknesses: {evaluation.get('weaknesses', [])}
        {targeted_clause}

        Instructions:
        - Directly challenge the candidate on their gap or incomplete reasoning.
        - Do NOT ask a generic question like "Can you explain more?".
        - Ask a concrete scenario, tradeoff, or edge case testing the exact weakness.
        """
        response = self._execute_with_telemetry(prompt, Question, task_type="generate_follow_up_question")
        return {**response["data"], "_metadata": response["metadata"]}

    def extract_skills(self, session_transcript: str) -> Dict[str, Any]:
        """Extract candidate skills from a session transcript."""
        prompt = f"""
        Analyze this interview session transcript and extract demonstrated skills:

        Transcript:
        {session_transcript}

        Identify both technical and soft skills, assign a score out of 100 for each, and provide evidence.
        """

        response = self._execute_with_telemetry(prompt, SkillExtraction, task_type="extract_skills")
        return {**response["data"], "_metadata": response["metadata"]}

    def synthesize_session(self, session_data: str) -> Dict[str, Any]:
        """Synthesize overall session results."""
        prompt = f"""
        Based on the following interview session data, synthesize an overall assessment:

        Data:
        {session_data}

        Provide an overall rating, a comprehensive summary, strengths, and red flags.
        """

        response = self._execute_with_telemetry(prompt, SessionSynthesis, task_type="synthesize_session")
        return {**response["data"], "_metadata": response["metadata"]}
