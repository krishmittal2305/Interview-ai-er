const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000/api'
import { getBrowserSupabaseClient } from './supabase'

export interface InterviewSession {
  id: string
  user_id: string
  interview_type: string
  start_time: string
  end_time?: string
  score: number
  status: 'active' | 'completed'
}

export interface Question {
  id: string
  session_id: string
  question_text: string
  interview_type: string
  answer_text?: string
  evaluation_score?: number
  evaluation_feedback?: string
  evaluation_details?: any
  code_text?: string
  programming_language?: string
  created_at?: string
  updated_at?: string
  difficulty?: string
  skill_focus?: string
  is_follow_up?: boolean
  parent_question_id?: string
  question_index?: number
  total_questions?: number
}

export interface CodeEvaluation {
  score: number
  feedback: string
  code_quality: number
  correctness: number
  efficiency?: number
  readability?: number
  algorithm_quality?: number
  complexity?: string
  time_complexity?: string
  space_complexity?: string
  edge_case_coverage?: number
  edge_cases_handled?: boolean
  testability?: number
  best_practices?: number
  strengths: string[]
  weaknesses?: string[]
  improvements?: string[]
  issues?: string[]
  evidence?: string
  ml_defect_detection?: {
    defect_probability: number
    risk_band: 'low' | 'medium' | 'high'
    model_version: string
    confidence: number
    inference_time_ms?: number
    method?: string
    risk_indicators?: string[]
    is_fine_tuned?: boolean
  }
}

export interface SessionIntegrityReport {
  review_recommended: boolean
  aggregation_score: number
  signals: Array<{
    signal_type: string
    timestamp: string
    source: string
    confidence: number
    evidence: string
    severity: 'LOW' | 'MEDIUM' | 'HIGH'
  }>
  summary: string
  confidence: number
  is_cheating?: boolean
  anomalies?: any[]
  risk_score?: number
}

export type SecurityCheck = any

export interface AnswerEvaluation {
  score: number
  overall_score?: number
  feedback: string
  improvements?: string[]
  weaknesses?: string[]
  strengths: string[]
  technical_accuracy?: number
  conceptual_depth?: number
  problem_solving?: number
  communication?: number
  completeness?: number
  evidence?: string
  recommended_follow_up?: string
}

export interface OrchestratorSessionState {
  session_id: string
  user_id: string
  target_role: string
  interview_type: string
  phase: string
  difficulty: string
  current_skill_focus?: string
  questions_count: number
  answered_count: number
  session_score: number
  final_score?: number | null
  weaknesses_discovered: string[]
  strengths_discovered: string[]
  pending_follow_ups: number
  time_budget_seconds: number
  start_time?: string
  skills_distribution: Record<string, {
    skill_name: string
    questions_count: number
    scores: number[]
    average_score: number
  }>
  is_completed: boolean
  final_assessment?: any
}

export interface CandidateSkillProfile {
  id?: string
  user_id: string
  skill_name: string
  estimated_proficiency: number
  confidence: 'insufficient evidence' | 'low confidence' | 'medium confidence' | 'high confidence'
  evidence_count: number
  recent_performance: number
  historical_performance: number
  improvement_trend: 'improving' | 'declining' | 'stable' | 'neutral'
  last_evaluated_timestamp?: string
  evidence_history?: Array<{
    session_id: string
    question_id: string
    score: number
    timestamp: string
  }>
}

export interface Recommendation {
  id: string
  user_id: string
  session_id?: string
  target_skill: string
  strategy: 'next_interview' | 'practice_session' | 'skill_reinforcement' | 'weak_signal_validation' | 'review_prior_mistakes'
  reason: string
  evidence: Record<string, any>
  recommended_activity: {
    type?: string
    title?: string
    description?: string
    difficulty?: string
    progression_stages?: string[]
    remediation_steps?: string[]
    focus_areas?: string[]
    topics?: string[]
    [key: string]: any
  }
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  expected_learning_objective: string
  status: 'PENDING' | 'ACCEPTED' | 'COMPLETED' | 'DISMISSED'
  created_at: string
  completed_at?: string
  baseline_proficiency: number
  post_outcome_proficiency?: number
  outcome_delta?: number
  outcome_assessment?: 'improved' | 'declined' | 'unchanged' | 'pending'
}

export interface PracticeSequenceItem {
  sequence_order: number
  question_id: string
  title: string
  question_text: string
  interview_type: string
  target_role: string
  skill_focus: string
  difficulty: string
  expected_time_minutes: number
  rubric: string
  canonical_skills: string[]
  selection_reason: string
  remediation_objective: string
  progression_stage: 'foundational_reinforcement' | 'scaffolded_progression' | 'frontier_mastery' | 'spaced_retention' | 'curriculum_breadth' | string
  model_metrics?: Record<string, any>
}

export interface PracticePlan {
  plan_id: string
  user_id: string
  mode: 'weakest_skills' | 'role_prep' | string
  target_role: string
  target_skills: string[]
  baseline_skill_estimates: Record<string, number>
  sequence: PracticeSequenceItem[]
  total_estimated_minutes: number
  summary_explanation: string
  created_at: string
}

export interface InferenceLogEvent {
  id: string
  timestamp: string
  inference_type: 'ml' | 'llm' | string
  model: string
  model_version: string
  task: string
  latency_ms: number
  success: boolean
  status: 'SUCCESS' | 'FAILED' | 'FALLBACK' | 'TIMEOUT' | string
  error_message?: string | null
  input_size: number
  output_size?: number | null
  input_tokens?: number | null
  output_tokens?: number | null
  memory_device?: string | null
  fallback_usage: boolean
  retry_count: number
  cost_estimate_usd?: number
  metadata?: Record<string, any>
}

export interface ModelUsageMetric {
  model: string
  inference_type: 'ml' | 'llm' | string
  total_calls: number
  share_pct: number
  avg_latency_ms: number
  p95_latency_ms: number
  failure_rate_pct: number
  fallback_rate_pct: number
}

export interface LLMUsageMetric {
  model: string
  total_calls: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  estimated_cost_usd: number
  avg_latency_ms: number
  p95_latency_ms: number
}

export interface TaskUsageMetric {
  task: string
  inference_type: 'ml' | 'llm' | string
  total_calls: number
  avg_latency_ms: number
  failure_rate_pct: number
}

export interface ObservabilityStats {
  time_window: string
  total_volume: number
  ml_volume: number
  llm_volume: number
  average_latency_ms: number
  p95_latency_ms: number
  failure_rate_pct: number
  fallback_rate_pct: number
  total_estimated_cost_usd: number
  model_usage: ModelUsageMetric[]
  llm_usage: LLMUsageMetric[]
  task_usage: TaskUsageMetric[]
  most_expensive_operations: InferenceLogEvent[]
}

export interface InferenceLogResponse {
  total: number
  limit: number
  offset: number
  events: InferenceLogEvent[]
}

class APIClient {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {}),
    }

    try {
      const supabase = getBrowserSupabaseClient()
      if (supabase) {
        const { data: { session } } = await supabase.auth.getSession()
        if (session?.access_token) {
          headers['Authorization'] = `Bearer ${session.access_token}`
        }
      }
    } catch (e) {
      console.warn("Failed to attach auth token", e)
    }
    
    const response = await fetch(url, {
      ...options,
      headers,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.message || `HTTP ${response.status}`)
    }

    return response.json()
  }

  // Interview Session Management
  async startSession(userId: string, interviewType: string): Promise<any> {
    const res: any = await this.request('/start_session', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, interview_type: interviewType }),
    })
    return res?.data ?? res
  }

  async getQuestion(sessionId: string, interviewType: string = 'Software Engineer', difficulty: string = 'intermediate'): Promise<any> {
    const params = new URLSearchParams({
      session_id: sessionId,
      interview_type: interviewType,
      difficulty,
    })
    const res: any = await this.request(`/get_question?${params}`)
    return res?.data ?? res
  }

  async submitAnswer(sessionId: string, questionId: string, answerText: string, inputModality: string = 'typed'): Promise<any> {
    const res: any = await this.request('/submit_answer', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        question_id: questionId,
        answer_text: answerText,
        input_modality: inputModality,
      }),
    })
    return res?.data ?? res
  }

  async submitVoiceAnswer(
    sessionId: string,
    questionId: string,
    transcript: string,
    transcriptionMetadata: Record<string, any> = {}
  ): Promise<any> {
    const res: any = await this.request('/submit_voice_answer', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        question_id: questionId,
        transcript,
        transcription_metadata: transcriptionMetadata,
      }),
    })
    return res?.data ?? res
  }

  async transcribeAudio(audioBlob: Blob): Promise<any> {
    const formData = new FormData()
    formData.append('file', audioBlob, 'recording.webm')

    let headers: Record<string, string> = {}
    try {
      const supabase = getBrowserSupabaseClient()
      if (supabase) {
        const { data } = await supabase.auth.getSession()
        if (data.session?.access_token) {
          headers['Authorization'] = `Bearer ${data.session.access_token}`
        }
      }
    } catch (e) {
      console.warn("Failed to attach auth token", e)
    }

    const response = await fetch(`${API_BASE_URL}/transcribe`, {
      method: 'POST',
      headers,
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.message || `HTTP ${response.status}`)
    }

    const res = await response.json()
    return res?.data ?? res
  }


  async submitCode(sessionId: string, questionId: string, code: string, language: string): Promise<any> {
    const res: any = await this.request('/submit_code', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        question_id: questionId,
        code,
        language,
      }),
    })
    return res?.data ?? res
  }

  async runCode(code: string, language: string): Promise<any> {
    const res: any = await this.request('/run_code', {
      method: 'POST',
      body: JSON.stringify({
        code,
        language,
      }),
    })
    return res?.data ?? res
  }

  async endSession(sessionId: string, finalScore?: number): Promise<any> {
    const res: any = await this.request(`/end_session/${sessionId}`, {
      method: 'POST',
      body: JSON.stringify({ final_score: finalScore }),
    })
    return res?.data ?? res
  }

  // Security & Monitoring
  async securityCheck(sessionId: string, securityData: any): Promise<SessionIntegrityReport> {
    const res: any = await this.request('/security/check', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        security_data: securityData,
      }),
    })
    return res?.data ?? res
  }

  async getSecurityReport(sessionId: string): Promise<any> {
    return this.request(`/security/report/${sessionId}`)
  }

  // Logging & Events
  async logEvent(sessionId: string, eventType: string, details: any = {}): Promise<{ timestamp: string }> {
    return this.request('/log_event', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        event_type: eventType,
        details,
      }),
    })
  }

  async logAnomaly(sessionId: string, anomalyType: string, severity: string, details: any = {}): Promise<void> {
    return this.request('/log_anomaly', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        anomaly_type: anomalyType,
        severity,
        details,
      }),
    })
  }

  // Session Information
  async getSessionDetails(sessionId: string): Promise<any> {
    const res: any = await this.request(`/session/${sessionId}`)
    return res?.data ?? res
  }

  async getSessionState(sessionId: string): Promise<OrchestratorSessionState> {
    const res: any = await this.request(`/session/${sessionId}/state`)
    return res?.data ?? res
  }

  async getUserSessions(userId: string, limit: number = 10): Promise<{ sessions: InterviewSession[]; total_count: number }> {
    return this.request(`/user/${userId}/sessions?limit=${limit}`)
  }

  // Practice & Adaptive Practice Features
  async generatePracticeSequence(
    mode: 'weakest_skills' | 'role_prep' = 'weakest_skills',
    targetRole: string = 'Software Engineer',
    numQuestions: number = 4
  ): Promise<PracticePlan> {
    const res: any = await this.request('/practice/sequence', {
      method: 'POST',
      body: JSON.stringify({
        mode,
        target_role: targetRole,
        num_questions: numQuestions,
      }),
    })
    return res?.data ?? res
  }

  async generatePracticeQuestion(
    interviewType: string = 'Software Engineer',
    difficulty?: string,
    topic?: string,
    mode: string = 'weakest_skills'
  ): Promise<any> {
    const res: any = await this.request('/practice/coding', {
      method: 'POST',
      body: JSON.stringify({
        interview_type: interviewType,
        difficulty,
        topic,
        mode,
      }),
    })
    return res?.data ?? res
  }

  async evaluatePracticeAnswer(
    question: string,
    answer: string,
    questionId?: string,
    topic?: string,
    difficulty?: string
  ): Promise<any> {
    const res: any = await this.request('/practice/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        question,
        answer,
        question_id: questionId,
        topic,
        difficulty,
      }),
    })
    return res?.data ?? res
  }

  async getFollowUpQuestion(question: string, answer: string, interviewType: string): Promise<{ follow_up_question: string }> {
    const res: any = await this.request('/follow_up_question', {
      method: 'POST',
      body: JSON.stringify({
        question,
        answer,
        interview_type: interviewType,
      }),
    })
    return res?.data ?? res
  }

  // Candidate Skill Intelligence & Recommendations
  async getCandidateSkills(userId: string): Promise<{ success: boolean; user_id: string; profiles: CandidateSkillProfile[] }> {
    return this.request(`/intelligence/skills/${userId}`)
  }

  async getRecommendations(userId: string, status?: string, strategy?: string): Promise<{ success: boolean; count: number; recommendations: Recommendation[] }> {
    const params = new URLSearchParams()
    if (status) params.append('status', status)
    if (strategy) params.append('strategy', strategy)
    const qs = params.toString() ? `?${params.toString()}` : ''
    return this.request(`/intelligence/recommendations/${userId}${qs}`)
  }

  async generateRecommendations(userId: string, sessionId?: string): Promise<{ success: boolean; count: number; recommendations: Recommendation[] }> {
    return this.request(`/intelligence/recommendations/${userId}/generate`, {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId })
    })
  }

  async updateRecommendationStatus(recommendationId: string, status: string): Promise<{ success: boolean; recommendation: Recommendation }> {
    return this.request(`/intelligence/recommendations/${recommendationId}/status`, {
      method: 'PATCH',
      body: JSON.stringify({ status })
    })
  }

  async evaluateRecommendationOutcome(recommendationId: string, currentProficiency?: number): Promise<{ success: boolean; outcome: any; recommendation: Recommendation }> {
    return this.request(`/intelligence/recommendations/${recommendationId}/evaluate_outcome`, {
      method: 'POST',
      body: JSON.stringify({ current_proficiency: currentProficiency })
    })
  }

  // Longitudinal Analytics
  async getLongitudinalAnalytics(userId: string, filters?: AnalyticsFilterParams): Promise<{ success: boolean; data: LongitudinalAnalyticsResponse }> {
    const params = new URLSearchParams()
    if (filters?.start_date) params.append('start_date', filters.start_date)
    if (filters?.end_date) params.append('end_date', filters.end_date)
    if (filters?.interview_type) params.append('interview_type', filters.interview_type)
    if (filters?.skill) params.append('skill', filters.skill)
    if (filters?.difficulty) params.append('difficulty', filters.difficulty)
    if (filters?.question_type) params.append('question_type', filters.question_type)
    const qs = params.toString() ? `?${params.toString()}` : ''
    return this.request(`/analytics/${userId}${qs}`)
  }

  async getAnalyticsFilterOptions(userId: string): Promise<{ success: boolean; data: AnalyticsFilterOptions }> {
    return this.request(`/analytics/filter-options/${userId}`)
  }

  // System Health
  async healthCheck(): Promise<{ status: string; database: string }> {
    return this.request('/health')
  }

  async getMetrics(): Promise<any> {
    return this.request('/metrics')
  }

  // Observability & Telemetry
  async getObservabilityStats(window: string = '24h', type: string = 'all'): Promise<ObservabilityStats> {
    const res: any = await this.request(`/observability/stats?window=${window}&type=${type}`)
    return res?.data ?? res
  }

  async getObservabilityEvents(params: {
    limit?: number
    offset?: number
    type?: string
    model?: string
    task?: string
    status?: string
    search?: string
  } = {}): Promise<InferenceLogResponse> {
    const q = new URLSearchParams()
    if (params.limit !== undefined) q.append('limit', String(params.limit))
    if (params.offset !== undefined) q.append('offset', String(params.offset))
    if (params.type) q.append('type', params.type)
    if (params.model) q.append('model', params.model)
    if (params.task) q.append('task', params.task)
    if (params.status) q.append('status', params.status)
    if (params.search) q.append('search', params.search)
    const qs = q.toString() ? `?${q.toString()}` : ''
    const res: any = await this.request(`/observability/events${qs}`)
    return res?.data ?? res
  }

  async triggerDiagnosticPing(target: string = 'ml'): Promise<any> {
    const res: any = await this.request('/observability/ping', {
      method: 'POST',
      body: JSON.stringify({ target }),
    })
    return res?.data ?? res
  }
}

export interface AnalyticsFilterParams {
  start_date?: string
  end_date?: string
  interview_type?: string
  skill?: string
  difficulty?: string
  question_type?: string
}

export interface ScoreTrendPoint {
  session_id: string
  session_index: number
  date: string
  interview_type: string
  raw_score: number
  difficulty_adjusted_score: number
  sma_3: number | null
  ema: number | null
  difficulty_level: string
  questions_count: number
  is_completed: boolean
}

export interface ScoreTrendSummary {
  data_points: ScoreTrendPoint[]
  linear_regression_slope: number
  r_squared: number
  trajectory_classification: 'significant_improvement' | 'moderate_improvement' | 'stable' | 'moderate_decline' | 'concerning_decline' | 'insufficient_data'
  baseline_score: number
  current_moving_average: number
  difficulty_adjusted_current: number
  total_sessions_analyzed: number
  methodology: string
}

export interface SkillTrendSummary {
  skill_name: string
  evidence_count: number
  estimated_proficiency: number
  decayed_proficiency: number
  recent_average: number
  historical_average: number
  net_delta: number
  volatility_sd: number
  confidence: 'insufficient evidence' | 'low confidence' | 'medium confidence' | 'high confidence'
  trend_status: 'demonstrated_growth' | 'plateaued' | 'skill_regression' | 'uncalibrated'
  last_evaluated: string | null
  score_history: Array<{
    score: number
    difficulty: string
    created_at?: string
  }>
}

export interface QuestionTypePerformance {
  question_type: string
  total_attempted: number
  average_score: number
  median_score: number
  pass_rate: number
  top_strengths: string[]
  top_weaknesses: string[]
}

export interface DifficultyLevelStats {
  level: string
  total_attempted: number
  average_score: number
  pass_rate: number
  standard_deviation: number
}

export interface DifficultyProgression {
  levels: Record<string, DifficultyLevelStats>
  promotion_transitions_attempted: number
  promotion_sustained_rate: number
  current_performance_frontier: 'beginner' | 'intermediate' | 'advanced'
  methodology: string
}

export interface ConsistencyAnalysis {
  score_count: number
  mean_score: number
  median_score: number
  standard_deviation: number
  coefficient_of_variation: number
  iqr: number
  min_score: number
  max_score: number
  score_range: number
  consistency_index: number
  consistency_category: 'Highly Consistent' | 'Moderately Consistent' | 'Volatile / High Variance' | 'Insufficient Data'
  methodology: string
}

export interface RepeatedWeakness {
  weakness_cluster: string
  canonical_label: string
  total_occurrences: number
  distinct_sessions_count: number
  session_percentage: number
  recency_flag: boolean
  persistence_status: 'persistent_blocker' | 'emerging_issue' | 'resolving' | 'sporadic'
  evidence_snippets: string[]
  target_skill?: string
}

export interface RecommendationImpact {
  recommendation_id: string
  target_skill: string
  strategy: string
  reason: string
  completed_at: string | null
  baseline_proficiency: number
  post_proficiency: number | null
  delta: number | null
  outcome_status: 'verified_improvement' | 'no_measurable_change' | 'regression' | 'awaiting_evidence'
  post_questions_evaluated: number
}

export interface RecentVsHistorical {
  recent_period_label: string
  historical_period_label: string
  recent_average_score: number
  historical_average_score: number
  score_delta: number
  score_pct_change: number
  recent_advanced_ratio: number
  historical_advanced_ratio: number
  recent_consistency_cv: number
  historical_consistency_cv: number
  recent_sample_size: number
  historical_sample_size: number
  statistically_meaningful: boolean
  verdict: string
}

export interface CompletionBehavior {
  total_sessions: number
  completed_sessions: number
  abandoned_sessions: number
  completion_rate: number
  avg_duration_minutes: number
  avg_questions_per_session: number
  completed_avg_score: number
  abandoned_avg_score: number
}

export interface NextPracticeRecommendation {
  priority: number
  target_skill: string
  recommended_difficulty: string
  question_type: string
  learning_objective: string
  rationale: string
  evidence_context: string
}

export interface SessionEvidenceAudit {
  session_id: string
  interview_type: string
  start_time: string
  score: number
  status: string
  questions: Array<{
    question_id: string
    question_text: string
    difficulty: string
    skill_focus: string
    score: number
    feedback: string
    is_code: boolean
  }>
}

export interface LongitudinalAnalyticsResponse {
  user_id: string
  generated_at: string
  filters_applied: AnalyticsFilterParams
  score_trends: ScoreTrendSummary
  skill_trends: SkillTrendSummary[]
  question_type_performance: QuestionTypePerformance[]
  difficulty_progression: DifficultyProgression
  consistency: ConsistencyAnalysis
  repeated_weaknesses: RepeatedWeakness[]
  recommendation_impact: RecommendationImpact[]
  recent_vs_historical: RecentVsHistorical
  completion_behavior: CompletionBehavior
  next_practice_recommendations: NextPracticeRecommendation[]
  session_evidence: SessionEvidenceAudit[]
}

export interface AnalyticsFilterOptions {
  interview_types: string[]
  skills: string[]
  difficulties: string[]
  question_types: string[]
  earliest_date: string | null
  latest_date: string | null
  total_sessions_count: number
  total_questions_count: number
}

export const apiClient = new APIClient()
