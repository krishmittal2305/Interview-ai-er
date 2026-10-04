import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from flask import current_app

logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for interacting with Supabase database (New Architecture)"""
    _shared_recommendations: Dict[str, Dict[str, Any]] = {}
    _shared_sessions: Dict[str, Dict[str, Any]] = {}
    _shared_questions: Dict[str, List[Dict[str, Any]]] = {}
    _shared_events: Dict[str, List[Dict[str, Any]]] = {}
    _shared_anomalies: Dict[str, List[Dict[str, Any]]] = {}
    _shared_skill_profiles: Dict[str, Dict[str, Any]] = {}
    _shared_question_skills: Dict[str, Dict[str, Any]] = {}
    _shared_inferences: Dict[str, Dict[str, Any]] = {}
    _shared_candidate_models: Dict[str, Dict[str, Any]] = {}
    _shared_question_embeddings: Dict[str, Dict[str, Any]] = {}
    _shared_inference_logs: List[Dict[str, Any]] = []
    _cache_file: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".local_storage_cache.json"
    )
    _cache_loaded: bool = False
    _remote_available: Optional[bool] = None

    def __init__(self):
        self.client: Optional[Client] = None
        self._local_recommendations = SupabaseService._shared_recommendations
        self._local_sessions = SupabaseService._shared_sessions
        self._local_questions = SupabaseService._shared_questions
        self._local_events = SupabaseService._shared_events
        self._local_anomalies = SupabaseService._shared_anomalies
        self._local_skill_profiles = SupabaseService._shared_skill_profiles
        self._local_question_skills = SupabaseService._shared_question_skills
        self._local_inferences = SupabaseService._shared_inferences
        self._local_candidate_models = SupabaseService._shared_candidate_models
        self._local_question_embeddings = SupabaseService._shared_question_embeddings
        self._local_inference_logs = SupabaseService._shared_inference_logs
        self._ensure_cache_loaded()

    @classmethod
    def _ensure_cache_loaded(cls):
        if cls._cache_loaded:
            return
        cls._cache_loaded = True
        try:
            if os.path.exists(cls._cache_file):
                with open(cls._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cls._shared_sessions.update(data.get("sessions", {}))
                    cls._shared_questions.update(data.get("questions", {}))
                    cls._shared_events.update(data.get("events", {}))
                    cls._shared_anomalies.update(data.get("anomalies", {}))
                    cls._shared_skill_profiles.update(data.get("skill_profiles", {}))
                    cls._shared_recommendations.update(data.get("recommendations", {}))
                    cls._shared_question_skills.update(data.get("question_skills", {}))
                    cls._shared_inferences.update(data.get("inferences", {}))
                    cls._shared_candidate_models.update(data.get("candidate_models", {}))
                    cls._shared_question_embeddings.update(data.get("question_embeddings", {}))
                    loaded_logs = data.get("inference_logs", [])
                    if loaded_logs and isinstance(loaded_logs, list):
                        cls._shared_inference_logs.extend(loaded_logs)
                logger.info(f"Loaded {len(cls._shared_sessions)} sessions from local storage cache.")
        except Exception as e:
            logger.warning(f"Failed to load local storage cache: {e}")

    @classmethod
    def _save_cache_to_disk(cls):
        try:
            # Keep up to latest 5000 inference logs in local cache to prevent unbounded file growth
            cache_data = {
                "sessions": cls._shared_sessions,
                "questions": cls._shared_questions,
                "events": cls._shared_events,
                "anomalies": cls._shared_anomalies,
                "skill_profiles": cls._shared_skill_profiles,
                "recommendations": cls._shared_recommendations,
                "question_skills": cls._shared_question_skills,
                "inferences": cls._shared_inferences,
                "candidate_models": cls._shared_candidate_models,
                "question_embeddings": cls._shared_question_embeddings,
                "inference_logs": cls._shared_inference_logs[-5000:],
            }
            tmp_file = cls._cache_file + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            if os.path.exists(cls._cache_file):
                os.replace(tmp_file, cls._cache_file)
            else:
                os.rename(tmp_file, cls._cache_file)
        except Exception as e:
            logger.warning(f"Failed to save local storage cache: {e}")

    def is_fallback_mode(self) -> bool:
        """Returns True if remote Supabase database tables are not accessible or configured"""
        return SupabaseService._remote_available is False

    def create_local_session(self, session_id: str, user_id: str, interview_type: str = "Software Engineer") -> Dict[str, Any]:
        """Creates or registers a local session (useful for state recovery or offline development)"""
        session_data = {
            'id': session_id,
            'user_id': user_id,
            'interview_type': interview_type,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'score': 0,
            'status': 'active'
        }
        self._local_sessions[session_id] = session_data
        self._save_cache_to_disk()
        logger.info(f"Local session registered/restored: {session_id} for user {user_id}")
        return session_data

    def _get_client(self) -> Client:
        """Lazy initialization of Supabase client with graceful context fallback."""
        if self.client is None:
            try:
                supabase_url = None
                supabase_key = None
                try:
                    if current_app:
                        supabase_url = current_app.config.get('SUPABASE_URL')
                        supabase_key = current_app.config.get('SUPABASE_KEY')
                except RuntimeError:
                    # Outside Flask application context
                    pass

                if not supabase_url:
                    supabase_url = os.environ.get('SUPABASE_URL')
                if not supabase_key:
                    supabase_key = os.environ.get('SUPABASE_KEY') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
                
                if not supabase_url or not supabase_key:
                    raise ValueError("Missing Supabase configuration (SUPABASE_URL / SUPABASE_KEY)")
                
                self.client = create_client(supabase_url, supabase_key)
                logger.info("Supabase client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}")
                raise
        return self.client

    def create_session(self, user_id: str, interview_type: str) -> Dict[str, Any]:
        """Create a new interview session"""
        session_data = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'interview_type': interview_type,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'score': 0,
            'status': 'active'
        }
        try:
            client = self._get_client()
            result = client.table('interview_sessions').insert(session_data).execute()
            if result.data:
                SupabaseService._remote_available = True
                session = result.data[0]
                self._local_sessions[session['id']] = session
                self._save_cache_to_disk()
                logger.info(f"Session created successfully: {session['id']}")
                return {
                    'success': True,
                    'session_id': session['id'],
                    'data': session
                }
            else:
                raise Exception("No data returned from session creation")
        except Exception as e:
            err_str = str(e)
            if "PGRST205" in err_str or "Could not find the table" in err_str or "interview_sessions" in err_str:
                SupabaseService._remote_available = False
                logger.warning(f"Supabase remote insert failed ({e}). Falling back to local in-memory session.")
                self._local_sessions[session_data['id']] = session_data
                self._save_cache_to_disk()
                return {
                    'success': True,
                    'session_id': session_data['id'],
                    'data': session_data
                }
            logger.error(f"Failed to create session: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        try:
            client = self._get_client()
            result = client.table('interview_sessions').select('*').eq('id', session_id).execute()
            if result.data:
                SupabaseService._remote_available = True
                return result.data[0]
            SupabaseService._remote_available = True
        except Exception as e:
            err_str = str(e)
            if "PGRST205" in err_str or "Could not find the table" in err_str or "interview_sessions" in err_str:
                SupabaseService._remote_available = False
            logger.warning(f"Failed to get session {session_id} from remote: {e}")
        return self._local_sessions.get(session_id)


    def update_session_score(self, session_id: str, new_score: float) -> bool:
        """Update session score"""
        if session_id in self._local_sessions:
            self._local_sessions[session_id]['score'] = new_score
            self._local_sessions[session_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
            self._save_cache_to_disk()
        try:
            client = self._get_client()
            result = client.table('interview_sessions').update({
                'score': new_score,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', session_id).execute()
            success = len(result.data) > 0
            if success:
                logger.info(f"Session {session_id} score updated to {new_score}")
            return success
        except Exception as e:
            logger.warning(f"Failed to update session score remotely: {e}")
            return session_id in self._local_sessions

    def end_session(self, session_id: str, final_score: float = None) -> bool:
        """End an interview session"""
        if session_id in self._local_sessions:
            self._local_sessions[session_id]['status'] = 'COMPLETED'
            self._local_sessions[session_id]['end_time'] = datetime.now(timezone.utc).isoformat()
            if final_score is not None:
                self._local_sessions[session_id]['score'] = final_score
            self._save_cache_to_disk()
        try:
            client = self._get_client()
            update_data = {
                'status': 'COMPLETED',
                'end_time': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            if final_score is not None:
                update_data['score'] = final_score
            result = client.table('interview_sessions').update(update_data).eq('id', session_id).execute()
            success = len(result.data) > 0
            if success:
                logger.info(f"Session {session_id} ended successfully")
            return success
        except Exception as e:
            logger.warning(f"Failed to end session {session_id} remotely: {e}")
            return session_id in self._local_sessions

    def store_question(self, session_id: str, question_text: str, interview_type: str = "technical") -> Optional[str]:
        """Store a question in the database"""
        question_data = {
            'id': str(uuid.uuid4()),
            'session_id': session_id,
            'question_text': question_text,
            'question_type': interview_type or 'technical',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        if session_id not in self._local_questions:
            self._local_questions[session_id] = []
        self._local_questions[session_id].append(question_data)
        self._save_cache_to_disk()

        try:
            client = self._get_client()
            result = client.table('interview_questions').insert(question_data).execute()
            if result.data:
                question_id = result.data[0]['id']
                logger.info(f"Question stored successfully: {question_id}")
                return question_id
        except Exception as e:
            logger.warning(f"Failed to store question remotely: {e}. Stored in local fallback cache.")
        return question_data['id']

    def store_answer(
        self,
        session_id: str,
        question_id: str,
        answer_text: str,
        evaluation: Dict[str, Any],
        transcription_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store an answer and its evaluation in normalized tables (with optional voice transcription metadata)."""
        # Extract model version attribution fields
        model_name = (
            evaluation.get('model_name')
            or evaluation.get('model_attribution', {}).get('model_name')
            or 'answer-nli-v1'
        )
        model_version = (
            evaluation.get('model_version')
            or evaluation.get('model_attribution', {}).get('model_version')
            or '1.0.0'
        )
        dataset_version = (
            evaluation.get('dataset_version')
            or evaluation.get('model_attribution', {}).get('dataset_version')
            or 'mnli-snli-v1'
        )
        training_run = (
            evaluation.get('training_run')
            or evaluation.get('model_attribution', {}).get('training_run')
            or 'answer-nli-v1_pretrained_deberta_minilm'
        )
        inference_timestamp = (
            evaluation.get('inference_timestamp')
            or evaluation.get('model_attribution', {}).get('inference_timestamp')
            or datetime.now(timezone.utc).isoformat()
        )

        evaluation['model_name'] = model_name
        evaluation['model_version'] = model_version
        evaluation['dataset_version'] = dataset_version
        evaluation['training_run'] = training_run
        evaluation['inference_timestamp'] = inference_timestamp

        if session_id in self._local_questions:
            for q in self._local_questions[session_id]:
                if q.get('id') == question_id:
                    # Immutability check: historical model results must NOT be overwritten!
                    if q.get('evaluation_details'):
                        if 'evaluation_history' not in q:
                            q['evaluation_history'] = [dict(q['evaluation_details'])]
                        q['evaluation_history'].append(dict(evaluation))
                        q['latest_evaluation'] = evaluation
                    else:
                        q['answer_text'] = answer_text
                        q['evaluation_score'] = evaluation.get('score', 0)
                        q['evaluation_feedback'] = evaluation.get('feedback', '')
                        q['evaluation_details'] = evaluation
                        q['evaluation_history'] = [dict(evaluation)]
                        q['model_name'] = model_name
                        q['model_version'] = model_version
                        q['dataset_version'] = dataset_version
                        q['training_run'] = training_run
                        q['inference_timestamp'] = inference_timestamp

                    if transcription_metadata:
                        q['transcript'] = answer_text
                        q['transcription_model'] = transcription_metadata.get('model_id') or transcription_metadata.get('transcription_model')
                        q['language'] = transcription_metadata.get('language')
                        q['duration'] = transcription_metadata.get('duration_seconds') or transcription_metadata.get('duration')
                        q['timestamp'] = transcription_metadata.get('timestamp')
                        q['transcription_metadata'] = transcription_metadata
            self._save_cache_to_disk()

        # Also store to dedicated model inferences tracking
        eval_id = str(uuid.uuid4())
        self.store_model_inference(
            model_name=model_name,
            model_version=model_version,
            dataset_version=dataset_version,
            training_run=training_run,
            inference_timestamp=inference_timestamp,
            inference_output=evaluation,
            session_id=session_id,
            question_id=question_id,
            evaluation_id=eval_id,
            source="ml",
        )

        try:
            client = self._get_client()
            response_id = str(uuid.uuid4())
            response_data = {
                'id': response_id,
                'question_id': question_id,
                'response_text': answer_text,
                'submitted_at': datetime.now(timezone.utc).isoformat()
            }
            if transcription_metadata:
                response_data['metadata'] = {
                    'transcript': answer_text,
                    'transcription_model': transcription_metadata.get('model_id') or transcription_metadata.get('transcription_model'),
                    'language': transcription_metadata.get('language'),
                    'duration': transcription_metadata.get('duration_seconds') or transcription_metadata.get('duration'),
                    'timestamp': transcription_metadata.get('timestamp'),
                    'confidence': transcription_metadata.get('confidence'),
                    'latency_ms': transcription_metadata.get('latency_ms'),
                }
            client.table('responses').insert(response_data).execute()
            
            eval_data = {
                'id': eval_id,
                'response_id': response_id,
                'score': evaluation.get('score', 0),
                'feedback': evaluation.get('feedback', ''),
                'evaluation_details': evaluation,
                'status': 'COMPLETED',
                'model_name': model_name,
                'model_version': model_version,
                'dataset_version': dataset_version,
                'training_run': training_run,
                'inference_timestamp': inference_timestamp,
            }
            client.table('evaluations').insert(eval_data).execute()
            logger.info(f"Answer stored successfully for question {question_id} (model: {model_name} v{model_version})")
            return True
        except Exception as e:
            logger.warning(f"Failed to store answer remotely: {e}. Stored in local fallback cache.")
            return True


    def store_code_submission(self, session_id: str, question_id: str, code: str, language: str, evaluation: Dict[str, Any]) -> bool:
        """Store code submission and evaluation in normalized tables"""
        # Extract model version attribution fields for code evaluation
        model_name = (
            evaluation.get('model_name')
            or evaluation.get('model_attribution', {}).get('model_name')
            or 'code-risk-v1'
        )
        model_version = (
            evaluation.get('model_version')
            or evaluation.get('model_attribution', {}).get('model_version')
            or '1.0.0'
        )
        dataset_version = (
            evaluation.get('dataset_version')
            or evaluation.get('model_attribution', {}).get('dataset_version')
            or 'CodeXGLUE-defect-v1'
        )
        training_run = (
            evaluation.get('training_run')
            or evaluation.get('model_attribution', {}).get('training_run')
            or 'code-risk-v1_codebert_ast_analysis'
        )
        inference_timestamp = (
            evaluation.get('inference_timestamp')
            or evaluation.get('model_attribution', {}).get('inference_timestamp')
            or datetime.now(timezone.utc).isoformat()
        )

        evaluation['model_name'] = model_name
        evaluation['model_version'] = model_version
        evaluation['dataset_version'] = dataset_version
        evaluation['training_run'] = training_run
        evaluation['inference_timestamp'] = inference_timestamp

        if session_id in self._local_questions:
            for q in self._local_questions[session_id]:
                if q.get('id') == question_id:
                    # Immutability check: historical code evaluations must NOT be overwritten!
                    if q.get('code_evaluation_details'):
                        if 'code_evaluation_history' not in q:
                            q['code_evaluation_history'] = [dict(q['code_evaluation_details'])]
                        q['code_evaluation_history'].append(dict(evaluation))
                        q['latest_code_evaluation'] = evaluation
                    else:
                        q['code_text'] = code
                        q['code_language'] = language
                        q['code_evaluation_score'] = evaluation.get('score', 0)
                        q['code_evaluation_feedback'] = evaluation.get('feedback', '')
                        q['code_evaluation_details'] = evaluation
                        q['code_evaluation_history'] = [dict(evaluation)]
                        q['code_model_name'] = model_name
                        q['code_model_version'] = model_version
                        q['code_dataset_version'] = dataset_version
                        q['code_training_run'] = training_run
                        q['code_inference_timestamp'] = inference_timestamp
            self._save_cache_to_disk()

        eval_id = str(uuid.uuid4())
        self.store_model_inference(
            model_name=model_name,
            model_version=model_version,
            dataset_version=dataset_version,
            training_run=training_run,
            inference_timestamp=inference_timestamp,
            inference_output=evaluation,
            session_id=session_id,
            question_id=question_id,
            evaluation_id=eval_id,
            source="ml",
        )

        try:
            client = self._get_client()
            
            # 1. Insert code submission
            submission_id = str(uuid.uuid4())
            code_data = {
                'id': submission_id,
                'question_id': question_id,
                'code_text': code,
                'programming_language': language,
                'submitted_at': datetime.now(timezone.utc).isoformat()
            }
            client.table('code_submissions').insert(code_data).execute()
            
            # 2. Insert execution run details
            exec_data = {
                'id': str(uuid.uuid4()),
                'code_submission_id': submission_id,
                'status': 'COMPLETED',
                'stdout': evaluation.get('stdout', ''),
                'stderr': evaluation.get('stderr', '')
            }
            client.table('execution_runs').insert(exec_data).execute()
            
            # Insert response + evaluation
            response_id = str(uuid.uuid4())
            client.table('responses').insert({
                'id': response_id,
                'question_id': question_id,
                'response_text': code,
                'submitted_at': datetime.now(timezone.utc).isoformat()
            }).execute()
            
            client.table('evaluations').insert({
                'id': eval_id,
                'response_id': response_id,
                'score': evaluation.get('score', 0),
                'feedback': evaluation.get('feedback', ''),
                'evaluation_details': evaluation,
                'status': 'COMPLETED',
                'model_name': model_name,
                'model_version': model_version,
                'dataset_version': dataset_version,
                'training_run': training_run,
                'inference_timestamp': inference_timestamp,
            }).execute()
            
            logger.info(f"Code submission stored successfully for question {question_id} (model: {model_name} v{model_version})")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to store code submission remotely: {e}. Stored in local fallback cache.")
            return True

    def get_session_questions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all questions for a session, including legacy format compatibility"""
        try:
            client = self._get_client()
            # Join questions with responses and evaluations
            result = client.table('interview_questions').select(
                '*, responses(*, evaluations(*))'
            ).eq('session_id', session_id).order('created_at').execute()
            
            if result.data:
                questions = []
                for q in result.data:
                    q_mapped = dict(q)
                    if q.get('responses') and len(q['responses']) > 0:
                        latest_response = q['responses'][-1]
                        q_mapped['answer_text'] = latest_response.get('response_text')
                        if latest_response.get('evaluations') and len(latest_response['evaluations']) > 0:
                            all_evals = latest_response['evaluations']
                            historical_eval = all_evals[0]
                            latest_eval = all_evals[-1]
                            q_mapped['evaluation_score'] = historical_eval.get('score')
                            q_mapped['evaluation_feedback'] = historical_eval.get('feedback')
                            q_mapped['evaluation_details'] = historical_eval.get('evaluation_details')
                            q_mapped['latest_evaluation'] = latest_eval.get('evaluation_details')
                            q_mapped['evaluation_history'] = [e.get('evaluation_details') or e for e in all_evals]
                            q_mapped['model_name'] = historical_eval.get('model_name') or (historical_eval.get('evaluation_details', {}) or {}).get('model_name')
                            q_mapped['model_version'] = historical_eval.get('model_version') or (historical_eval.get('evaluation_details', {}) or {}).get('model_version')
                            q_mapped['dataset_version'] = historical_eval.get('dataset_version') or (historical_eval.get('evaluation_details', {}) or {}).get('dataset_version')
                            q_mapped['training_run'] = historical_eval.get('training_run') or (historical_eval.get('evaluation_details', {}) or {}).get('training_run')
                            q_mapped['inference_timestamp'] = historical_eval.get('inference_timestamp') or (historical_eval.get('evaluation_details', {}) or {}).get('inference_timestamp')
                    questions.append(q_mapped)
                return questions
        except Exception as e:
            logger.warning(f"Failed to get questions remotely for session {session_id}: {e}")
        return self._local_questions.get(session_id, [])

    def log_event(self, session_id: str, event_type: str, details: Dict[str, Any] = None) -> bool:
        """Log a system event to session_events"""
        event_data = {
            'id': str(uuid.uuid4()),
            'session_id': session_id,
            'event_type': event_type,
            'event_data': details or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        if session_id not in self._local_events:
            self._local_events[session_id] = []
        self._local_events[session_id].append(event_data)
        self._save_cache_to_disk()
        try:
            client = self._get_client()
            result = client.table('session_events').insert(event_data).execute()
            return len(result.data) > 0
        except Exception as e:
            return True

    def log_anomaly(self, session_id: str, anomaly_type: str, severity: str, details: Dict[str, Any] = None) -> bool:
        """Log suspicious behavior to integrity_events"""
        anomaly_data = {
            'id': str(uuid.uuid4()),
            'session_id': session_id,
            'event_type': anomaly_type,
            'severity': severity,
            'evidence_details': details or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        if session_id not in self._local_anomalies:
            self._local_anomalies[session_id] = []
        self._local_anomalies[session_id].append(anomaly_data)
        self._save_cache_to_disk()
        try:
            client = self._get_client()
            result = client.table('integrity_events').insert(anomaly_data).execute()
            return len(result.data) > 0
        except Exception as e:
            return True

    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions for a user"""
        try:
            client = self._get_client()
            result = client.table('interview_sessions').select('*').eq('user_id', user_id).order('start_time', desc=True).limit(limit).execute()
            if result.data:
                return result.data
        except Exception as e:
            logger.warning(f"Failed to get sessions remotely for user {user_id}: {e}")
        local_user_sessions = [s for s in self._local_sessions.values() if s.get('user_id') == user_id]
        return sorted(local_user_sessions, key=lambda s: s.get('start_time', ''), reverse=True)[:limit]

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a session with full local fallback"""
        session = None
        try:
            client = self._get_client()
            session_result = client.table('interview_sessions').select('*').eq('id', session_id).execute()
            if session_result.data:
                session = session_result.data[0]
        except Exception as e:
            logger.warning(f"Failed to get session remotely in get_session_statistics: {e}")

        if not session:
            session = self._local_sessions.get(session_id)

        if not session:
            return {}

        questions = self.get_session_questions(session_id)
        total_questions = len(questions)
        answered_questions = len([q for q in questions if q.get('answer_text') or q.get('code_text')])
        avg_score = 0
        if answered_questions > 0:
            scores = [q.get('evaluation_score') or q.get('code_evaluation_score', 0) for q in questions if q.get('evaluation_score') or q.get('code_evaluation_score')]
            avg_score = sum(scores) / len(scores) if scores else 0

        events = []
        try:
            client = self._get_client()
            events_result = client.table('session_events').select('*').eq('session_id', session_id).execute()
            events = events_result.data if events_result.data else []
        except Exception:
            events = self._local_events.get(session_id, [])

        anomalies = []
        try:
            client = self._get_client()
            anomalies_result = client.table('integrity_events').select('*').eq('session_id', session_id).execute()
            anomalies = anomalies_result.data if anomalies_result.data else []
        except Exception:
            anomalies = self._local_anomalies.get(session_id, [])

        return {
            'session': session,
            'statistics': {
                'total_questions': total_questions,
                'answered_questions': answered_questions,
                'completion_rate': (answered_questions / total_questions * 100) if total_questions > 0 else 0,
                'average_score': round(avg_score, 2),
                'total_events': len(events),
                'total_anomalies': len(anomalies)
            },
            'questions': questions,
            'events': events,
            'anomalies': anomalies
        }


    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old completed sessions"""
        try:
            client = self._get_client()
            cutoff_date = datetime.now(timezone.utc).replace(tzinfo=timezone.utc) - timedelta(days=days_old)
            
            old_sessions = client.table('interview_sessions').select('id').lt('start_time', cutoff_date.isoformat()).eq('status', 'COMPLETED').execute()
            
            if not old_sessions.data:
                return 0
            
            deleted_count = 0
            for session in old_sessions.data:
                try:
                    # RLS and ON DELETE CASCADE should handle the rest
                    client.table('interview_sessions').delete().eq('id', session['id']).execute()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to cleanup session {session['id']}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old sessions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0

    def get_candidate_skill_profile(self, user_id: str, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get candidate skill profile"""
        key = f"{user_id}:{skill_name}"
        try:
            client = self._get_client()
            result = client.table('candidate_skill_profiles').select('*').eq('user_id', user_id).eq('skill_name', skill_name).execute()
            if result.data:
                return result.data[0]
        except Exception as e:
            logger.warning(f"Failed to get skill profile remotely: {e}")
        return self._local_skill_profiles.get(key)

    def update_candidate_skill_profile(self, user_id: str, skill_name: str, profile_data: Dict[str, Any]) -> bool:
        """Update or create a candidate skill profile"""
        key = f"{user_id}:{skill_name}"
        data_to_store = dict(profile_data)
        data_to_store['id'] = self._local_skill_profiles.get(key, {}).get('id') or str(uuid.uuid4())
        data_to_store['user_id'] = user_id
        data_to_store['skill_name'] = skill_name
        data_to_store['updated_at'] = datetime.now(timezone.utc).isoformat()
        self._local_skill_profiles[key] = data_to_store
        self._save_cache_to_disk()
        try:
            client = self._get_client()
            existing = self.get_candidate_skill_profile(user_id, skill_name)
            if existing and existing.get('id') != data_to_store['id']:
                result = client.table('candidate_skill_profiles').update(data_to_store).eq('id', existing['id']).execute()
            else:
                result = client.table('candidate_skill_profiles').insert(data_to_store).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.warning(f"Failed to update skill profile remotely: {e}. Stored locally.")
            return True

    def get_user_skill_profiles(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all skill profiles for a user"""
        try:
            client = self._get_client()
            result = client.table('candidate_skill_profiles').select('*').eq('user_id', user_id).execute()
            if result.data:
                return result.data
        except Exception as e:
            logger.warning(f"Failed to get skill profiles remotely: {e}")
        return [p for p in self._local_skill_profiles.values() if p.get('user_id') == user_id]

    def get_candidate_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the full candidate model."""
        try:
            client = self._get_client()
            result = client.table('candidate_models').select('*').eq('user_id', user_id).execute()
            if result.data:
                return result.data[0]
        except Exception as e:
            pass
        return self._local_candidate_models.get(user_id)
        
    def save_candidate_model(self, user_id: str, model_data: Dict[str, Any]) -> bool:
        """Save the candidate model."""
        self._local_candidate_models[user_id] = model_data
        self._save_cache_to_disk()
        try:
            client = self._get_client()
            existing = self.get_candidate_model(user_id)
            if existing:
                client.table('candidate_models').update(model_data).eq('user_id', user_id).execute()
            else:
                client.table('candidate_models').insert(model_data).execute()
            return True
        except Exception as e:
            return True

    def create_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist recommendations to Supabase with in-memory fallback"""
        persisted = []
        for rec in recommendations:
            rec_data = dict(rec)
            rec_id = rec_data.get('id') or str(uuid.uuid4())
            rec_data['id'] = rec_id
            rec_data['user_id'] = user_id
            if 'created_at' not in rec_data:
                rec_data['created_at'] = datetime.now(timezone.utc).isoformat()
            
            if hasattr(rec_data.get('status'), 'value'):
                rec_data['status'] = rec_data['status'].value
            if hasattr(rec_data.get('strategy'), 'value'):
                rec_data['strategy'] = rec_data['strategy'].value
            if hasattr(rec_data.get('priority'), 'value'):
                rec_data['priority'] = rec_data['priority'].value
            
            self._local_recommendations[rec_id] = rec_data
            
            try:
                client = self._get_client()
                result = client.table('recommendations').insert(rec_data).execute()
                if result.data:
                    persisted.append(result.data[0])
                else:
                    persisted.append(rec_data)
            except Exception as e:
                logger.warning(f"Supabase remote insert failed: {e}")
                persisted.append(rec_data)
                
        self._save_cache_to_disk()
        return persisted

    def get_recommendations(self, user_id: str, status: Optional[str] = None, strategy: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recommendations for a user"""
        try:
            client = self._get_client()
            query = client.table('recommendations').select('*').eq('user_id', user_id)
            if status:
                query = query.eq('status', status.upper())
            if strategy:
                query = query.eq('strategy', strategy)
            result = query.order('created_at', desc=True).limit(limit).execute()
            if result.data:
                return result.data
        except Exception as e:
            logger.warning(f"Failed to query remote table: {e}. Falling back to local cache.")

        results = [r for r in self._local_recommendations.values() if r.get('user_id') == user_id]
        if status:
            target_status = status.upper()
            results = [r for r in results if str(r.get('status', '')).upper() == target_status]
        if strategy:
            target_strat = str(strategy).lower()
            results = [r for r in results if str(r.get('strategy', '')).lower() == target_strat]

        results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return results[:limit]

    def get_recommendation(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific recommendation by ID"""
        try:
            client = self._get_client()
            result = client.table('recommendations').select('*').eq('id', recommendation_id).execute()
            if result.data:
                return result.data[0]
        except Exception as e:
            logger.warning(f"Failed to fetch remote recommendation: {e}")

        return self._local_recommendations.get(recommendation_id)

    def update_recommendation(self, recommendation_id: str, update_data: Dict[str, Any]) -> bool:
        """Update recommendation status and outcome data"""
        data = dict(update_data)
        data['updated_at'] = datetime.now(timezone.utc).isoformat()

        if recommendation_id in self._local_recommendations:
            self._local_recommendations[recommendation_id].update(data)
            self._save_cache_to_disk()

        try:
            client = self._get_client()
            result = client.table('recommendations').update(data).eq('id', recommendation_id).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.warning(f"Failed to update remote recommendation: {e}")
            return recommendation_id in self._local_recommendations

    def health_check(self) -> Dict[str, Any]:
        """Check database connectivity and health"""
        try:
            client = self._get_client()
            result = client.table('interview_sessions').select('id').limit(1).execute()
            
            return {
                'status': 'healthy',
                'database': 'connected',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'database': 'disconnected',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def store_question_skill_prediction(
        self,
        question_id: str,
        predicted_skills: List[Dict[str, Any]],
        confidence: float,
        model_version: str,
        timestamp: Optional[str] = None,
        model_name: str = "question-skill-v1",
        dataset_version: str = "2026.10",
        training_run: str = "question-skill-tagger-v1_20261003T063901Z",
        inference_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Persist ML-derived skill metadata for a question.
        Maintains strict separation between original question metadata and ML predictions.
        Stamps model_name, model_version, dataset_version, training_run, and inference_timestamp.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        inf_ts = inference_timestamp or timestamp or now_iso
        record = {
            "id": str(uuid.uuid4()),
            "question_id": question_id,
            "predicted_skills": predicted_skills,
            "confidence": float(confidence),
            "model_version": model_version,
            "model_name": model_name,
            "dataset_version": dataset_version,
            "training_run": training_run,
            "inference_timestamp": inf_ts,
            "timestamp": timestamp or now_iso,
            "created_at": now_iso
        }
        self._local_question_skills[question_id] = record
        self._save_cache_to_disk()

        # Dedicated model inference tracking
        self.store_model_inference(
            model_name=model_name,
            model_version=model_version,
            dataset_version=dataset_version,
            training_run=training_run,
            inference_timestamp=inf_ts,
            inference_output={"predicted_skills": predicted_skills, "confidence": confidence},
            question_id=question_id,
            source="ml",
        )

        try:
            client = self._get_client()
            result = client.table("question_skill_predictions").insert(record).execute()
            if result.data:
                logger.info(f"Question skill prediction stored remotely for question {question_id} (model: {model_name} v{model_version})")
                return result.data[0]
        except Exception as e:
            logger.warning(f"Failed to store question skill prediction remotely: {e}. Stored in local fallback cache.")

        return record

    def get_question_skill_prediction(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve ML-derived skill metadata for a question."""
        try:
            client = self._get_client()
            result = client.table("question_skill_predictions").select("*").eq("question_id", question_id).execute()
            if result.data:
                return result.data[0]
        except Exception as e:
            logger.debug(f"Failed to fetch question skill prediction remotely: {e}")

        return self._local_question_skills.get(question_id)

    def store_model_inference(
        self,
        model_name: str,
        model_version: str,
        dataset_version: str,
        training_run: str,
        inference_timestamp: str,
        inference_output: Dict[str, Any],
        session_id: Optional[str] = None,
        question_id: Optional[str] = None,
        response_id: Optional[str] = None,
        evaluation_id: Optional[str] = None,
        source: str = "ml",
    ) -> Dict[str, Any]:
        """
        Store an immutable inference record attributable to a model version.
        Guarantees that every inference in the database has an identifiable lineage.
        """
        inference_id = str(uuid.uuid4())
        record = {
            "id": inference_id,
            "session_id": session_id,
            "question_id": question_id,
            "response_id": response_id,
            "evaluation_id": evaluation_id,
            "model_name": model_name,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "training_run": training_run,
            "inference_timestamp": inference_timestamp,
            "inference_output": inference_output,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._local_inferences[inference_id] = record
        self._save_cache_to_disk()

        try:
            client = self._get_client()
            result = client.table("model_inferences").insert(record).execute()
            if result.data:
                return result.data[0]
        except Exception as e:
            logger.debug(f"Failed to store model inference remotely: {e}")

        return record

    def get_model_inferences(
        self,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve model inferences matching filters."""
        try:
            client = self._get_client()
            query = client.table("model_inferences").select("*")
            if model_name:
                query = query.eq("model_name", model_name)
            if session_id:
                query = query.eq("session_id", session_id)
            if question_id:
                query = query.eq("question_id", question_id)
            res = query.order("inference_timestamp").execute()
            if res.data:
                return res.data
        except Exception as e:
            logger.debug(f"Failed to query model inferences remotely: {e}")

        inferences = list(self._local_inferences.values())
        if model_name:
            inferences = [inf for inf in inferences if inf.get("model_name") == model_name]
        if session_id:
            inferences = [inf for inf in inferences if inf.get("session_id") == session_id]
        if question_id:
            inferences = [inf for inf in inferences if inf.get("question_id") == question_id]
        return inferences

    def get_historical_evaluation(self, session_id: str, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the original historical evaluation for a question in a session.
        Guarantees that the evaluation retains its original model version regardless of newer models deployed.
        """
        questions = self.get_session_questions(session_id)
        for q in questions:
            if q.get("id") == question_id:
                # Return original evaluation_details or first item in evaluation_history
                if q.get("evaluation_history") and len(q["evaluation_history"]) > 0:
                    return q["evaluation_history"][0]
                if q.get("code_evaluation_history") and len(q["code_evaluation_history"]) > 0:
                    return q["code_evaluation_history"][0]
                return q.get("evaluation_details") or q.get("code_evaluation_details")
        return None

    def save_question_embedding(
        self,
        question_id: str,
        embedding_vector: List[float],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_version: str = "all-minilm-l6-v2",
        question_text: str = "",
        skill_focus: str = "",
        difficulty: str = "",
    ) -> Dict[str, Any]:
        """
        Store a question embedding vector in Supabase/PostgreSQL with local fallback cache.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        record = {
            "question_id": question_id,
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
            "embedding_vector": embedding_vector,
            "question_text": question_text,
            "skill_focus": skill_focus,
            "difficulty": difficulty,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        self._local_question_embeddings[question_id] = record
        self._save_cache_to_disk()

        try:
            client = self._get_client()
            res = client.table("question_embeddings").upsert(record, on_conflict="question_id").execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            logger.debug(f"Failed to persist question embedding remotely: {e}. Stored in local fallback cache.")

        return record

    def get_question_embedding(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve question embedding metadata and vector for a question."""
        try:
            client = self._get_client()
            res = client.table("question_embeddings").select("*").eq("question_id", question_id).execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            logger.debug(f"Failed to query question embedding remotely: {e}")

        return self._local_question_embeddings.get(question_id)

    def get_all_question_embeddings(
        self,
        embedding_model: Optional[str] = None,
        embedding_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all stored question embeddings."""
        try:
            client = self._get_client()
            query = client.table("question_embeddings").select("*")
            if embedding_model:
                query = query.eq("embedding_model", embedding_model)
            if embedding_version:
                query = query.eq("embedding_version", embedding_version)
            res = query.execute()
            if res.data and len(res.data) > 0:
                # Update local cache with remote records
                for r in res.data:
                    self._local_question_embeddings[r["question_id"]] = r
                return res.data
        except Exception as e:
            logger.debug(f"Failed to fetch question embeddings remotely: {e}")

        records = list(self._local_question_embeddings.values())
        if embedding_model:
            records = [r for r in records if r.get("embedding_model") == embedding_model]
        if embedding_version:
            records = [r for r in records if r.get("embedding_version") == embedding_version]
        return records

    def save_inference_log(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persist an AI/ML inference telemetry span into Supabase/PostgreSQL with local disk cache.
        """
        # Ensure ID and timestamp
        if not record.get("id"):
            record["id"] = str(uuid.uuid4())
        if not record.get("timestamp"):
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add to local in-memory log list (FIFO capped)
        self._local_inference_logs.append(record)
        if len(self._local_inference_logs) > 5000:
            self._local_inference_logs.pop(0)
        self._save_cache_to_disk()

        try:
            client = self._get_client()
            db_record = {
                "id": record["id"],
                "timestamp": record["timestamp"],
                "inference_type": record.get("inference_type", "ml"),
                "model": record.get("model", "unknown"),
                "model_version": record.get("model_version", "v1.0"),
                "task": record.get("task", "unknown"),
                "latency_ms": record.get("latency_ms", 0.0),
                "success": record.get("success", True),
                "status": record.get("status", "SUCCESS"),
                "error_message": record.get("error_message"),
                "input_size": record.get("input_size", 0),
                "output_size": record.get("output_size"),
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
                "memory_device": record.get("memory_device", "cpu"),
                "fallback_usage": record.get("fallback_usage", False),
                "retry_count": record.get("retry_count", 0),
                "cost_estimate_usd": record.get("cost_estimate_usd", 0.0),
                "metadata": record.get("metadata", {}),
            }
            client.table("ai_inference_logs").insert(db_record).execute()
        except Exception as e:
            logger.debug(f"Remote inference log persistence skipped: {e}. Persisted locally.")

        return record

    def get_inference_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        inference_type: Optional[str] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve paginated inference telemetry records with filtering.
        """
        # Read from local cache (which contains latest recorded runs)
        logs = list(self._local_inference_logs)

        # Apply filters
        if inference_type and inference_type.lower() != "all":
            logs = [l for l in logs if l.get("inference_type", "").lower() == inference_type.lower()]
        if model and model.lower() != "all":
            logs = [l for l in logs if model.lower() in l.get("model", "").lower()]
        if task and task.lower() != "all":
            logs = [l for l in logs if task.lower() in l.get("task", "").lower()]
        if status and status.lower() != "all":
            logs = [l for l in logs if l.get("status", "").upper() == status.upper()]
        if search:
            q = search.lower()
            logs = [
                l for l in logs
                if q in l.get("model", "").lower()
                or q in l.get("task", "").lower()
                or q in str(l.get("error_message", "")).lower()
                or q in l.get("memory_device", "").lower()
            ]

        # Sort reverse chronological
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        total_count = len(logs)
        paginated = logs[offset : offset + limit]

        return {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "events": paginated,
        }

    def get_observability_stats(
        self,
        time_window: Optional[str] = "24h",
        inference_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates genuine, non-fabricated observability metrics across recorded ML & LLM operations:
        - ML inference volume
        - Average latency
        - P95 latency
        - Failure rate
        - Fallback rate
        - Model usage breakdown
        - LLM usage breakdown
        - Most expensive operations
        """
        logs = list(self._local_inference_logs)

        # Filter by time window if specified
        now = datetime.now(timezone.utc)
        cutoff: Optional[datetime] = None
        if time_window == "1h":
            cutoff = now - timedelta(hours=1)
        elif time_window == "24h":
            cutoff = now - timedelta(days=1)
        elif time_window == "7d":
            cutoff = now - timedelta(days=7)
        elif time_window == "30d":
            cutoff = now - timedelta(days=30)

        if cutoff:
            filtered_by_time = []
            for l in logs:
                ts_str = l.get("timestamp")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts >= cutoff:
                            filtered_by_time.append(l)
                    except Exception:
                        filtered_by_time.append(l)
                else:
                    filtered_by_time.append(l)
            logs = filtered_by_time

        if inference_type and inference_type.lower() != "all":
            logs = [l for l in logs if l.get("inference_type", "").lower() == inference_type.lower()]

        total_volume = len(logs)

        # Zero-state when no data recorded yet (do NOT fabricate numbers)
        if total_volume == 0:
            return {
                "time_window": time_window,
                "total_volume": 0,
                "ml_volume": 0,
                "llm_volume": 0,
                "average_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "failure_rate_pct": 0.0,
                "fallback_rate_pct": 0.0,
                "total_estimated_cost_usd": 0.0,
                "model_usage": [],
                "llm_usage": [],
                "task_usage": [],
                "most_expensive_operations": [],
            }

        ml_volume = sum(1 for l in logs if l.get("inference_type", "").lower() == "ml")
        llm_volume = sum(1 for l in logs if l.get("inference_type", "").lower() == "llm")
        failures = sum(1 for l in logs if not l.get("success", True) or l.get("status", "").upper() == "FAILED")
        fallbacks = sum(1 for l in logs if l.get("fallback_usage") is True or l.get("status", "").upper() == "FALLBACK")

        latencies = [float(l.get("latency_ms", 0.0)) for l in logs]
        avg_latency = round(sum(latencies) / total_volume, 2) if latencies else 0.0

        sorted_latencies = sorted(latencies)
        p95_idx = int(round(0.95 * (len(sorted_latencies) - 1)))
        p95_latency = round(sorted_latencies[p95_idx], 2) if sorted_latencies else 0.0

        failure_rate_pct = round((failures / total_volume) * 100.0, 2)
        fallback_rate_pct = round((fallbacks / total_volume) * 100.0, 2)
        total_cost = round(sum(float(l.get("cost_estimate_usd", 0.0) or 0.0) for l in logs), 6)

        # Model usage breakdown
        model_groups: Dict[str, List[Dict[str, Any]]] = {}
        for l in logs:
            m = l.get("model", "unknown")
            model_groups.setdefault(m, []).append(l)

        model_usage = []
        for m_name, m_logs in model_groups.items():
            m_count = len(m_logs)
            m_lats = sorted([float(x.get("latency_ms", 0.0)) for x in m_logs])
            m_avg = round(sum(m_lats) / m_count, 2)
            m_p95_idx = int(round(0.95 * (len(m_lats) - 1)))
            m_p95 = round(m_lats[m_p95_idx], 2)
            m_fails = sum(1 for x in m_logs if not x.get("success", True) or x.get("status", "").upper() == "FAILED")
            m_falls = sum(1 for x in m_logs if x.get("fallback_usage") is True)
            m_type = m_logs[0].get("inference_type", "ml")

            model_usage.append({
                "model": m_name,
                "inference_type": m_type,
                "total_calls": m_count,
                "share_pct": round((m_count / total_volume) * 100.0, 1),
                "avg_latency_ms": m_avg,
                "p95_latency_ms": m_p95,
                "failure_rate_pct": round((m_fails / m_count) * 100.0, 2),
                "fallback_rate_pct": round((m_falls / m_count) * 100.0, 2),
            })
        model_usage.sort(key=lambda x: x["total_calls"], reverse=True)

        # LLM specific usage breakdown
        llm_logs = [l for l in logs if l.get("inference_type", "").lower() == "llm"]
        llm_groups: Dict[str, List[Dict[str, Any]]] = {}
        for l in llm_logs:
            m = l.get("model", "unknown")
            llm_groups.setdefault(m, []).append(l)

        llm_usage = []
        for m_name, m_logs in llm_groups.items():
            m_count = len(m_logs)
            in_tok = sum(int(x.get("input_tokens") or 0) for x in m_logs)
            out_tok = sum(int(x.get("output_tokens") or 0) for x in m_logs)
            cost = round(sum(float(x.get("cost_estimate_usd") or 0.0) for x in m_logs), 6)
            m_lats = sorted([float(x.get("latency_ms", 0.0)) for x in m_logs])
            m_avg = round(sum(m_lats) / m_count, 2)
            m_p95_idx = int(round(0.95 * (len(m_lats) - 1)))
            m_p95 = round(m_lats[m_p95_idx], 2)

            llm_usage.append({
                "model": m_name,
                "total_calls": m_count,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "estimated_cost_usd": cost,
                "avg_latency_ms": m_avg,
                "p95_latency_ms": m_p95,
            })
        llm_usage.sort(key=lambda x: x["total_calls"], reverse=True)

        # Task usage breakdown
        task_groups: Dict[str, List[Dict[str, Any]]] = {}
        for l in logs:
            t = l.get("task", "unknown")
            task_groups.setdefault(t, []).append(l)

        task_usage = []
        for t_name, t_logs in task_groups.items():
            t_count = len(t_logs)
            t_lats = [float(x.get("latency_ms", 0.0)) for x in t_logs]
            t_avg = round(sum(t_lats) / t_count, 2)
            t_fails = sum(1 for x in t_logs if not x.get("success", True) or x.get("status", "").upper() == "FAILED")

            task_usage.append({
                "task": t_name,
                "inference_type": t_logs[0].get("inference_type", "ml"),
                "total_calls": t_count,
                "avg_latency_ms": t_avg,
                "failure_rate_pct": round((t_fails / t_count) * 100.0, 2),
            })
        task_usage.sort(key=lambda x: x["total_calls"], reverse=True)

        # Most expensive operations (sorted by latency and cost)
        sorted_by_expense = sorted(
            logs,
            key=lambda x: (float(x.get("cost_estimate_usd", 0.0) or 0.0), float(x.get("latency_ms", 0.0))),
            reverse=True
        )
        most_expensive_operations = sorted_by_expense[:15]

        return {
            "time_window": time_window,
            "total_volume": total_volume,
            "ml_volume": ml_volume,
            "llm_volume": llm_volume,
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "failure_rate_pct": failure_rate_pct,
            "fallback_rate_pct": fallback_rate_pct,
            "total_estimated_cost_usd": total_cost,
            "model_usage": model_usage,
            "llm_usage": llm_usage,
            "task_usage": task_usage,
            "most_expensive_operations": most_expensive_operations,
        }
