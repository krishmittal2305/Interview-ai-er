-- Authoritative Schema Migration (Clean & Idempotent)
-- Consolidates platform data model into a clean, normalized relational model.

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. PROFILES (Extend auth.users)
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Handle existing sessions table -> interview_sessions (only if in public schema)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sessions') AND 
       NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'interview_sessions') THEN
        EXECUTE 'ALTER TABLE public.sessions RENAME TO interview_sessions';
        BEGIN
            EXECUTE 'ALTER TABLE public.interview_sessions RENAME CONSTRAINT sessions_pkey TO interview_sessions_pkey';
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;
    END IF;
END $$;

-- 2. INTERVIEW_SESSIONS
CREATE TABLE IF NOT EXISTS interview_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    interview_type VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    score DECIMAL(5,2) DEFAULT 0.00,
    session_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE interview_sessions ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_interview_sessions_user_id ON interview_sessions(user_id);

-- Handle existing questions table -> interview_questions (only if in public schema)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'questions') AND 
       NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'interview_questions') THEN
        EXECUTE 'ALTER TABLE public.questions RENAME TO interview_questions';
        BEGIN
            EXECUTE 'ALTER TABLE public.interview_questions RENAME CONSTRAINT questions_pkey TO interview_questions_pkey';
        EXCEPTION WHEN OTHERS THEN
            NULL;
        END;
    END IF;
END $$;

-- 3. INTERVIEW_QUESTIONS
CREATE TABLE IF NOT EXISTS interview_questions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    question_type VARCHAR(100) DEFAULT 'technical',
    difficulty VARCHAR(20) DEFAULT 'medium',
    expected_format VARCHAR(100),
    question_order INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE interview_questions ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_interview_questions_session_id ON interview_questions(session_id);

-- 4. RESPONSES
CREATE TABLE IF NOT EXISTS responses (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    question_id UUID REFERENCES interview_questions(id) ON DELETE CASCADE,
    response_text TEXT,
    audio_url VARCHAR(255),
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE responses ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_responses_question_id ON responses(question_id);

-- 5. EVALUATIONS
CREATE TABLE IF NOT EXISTS evaluations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    response_id UUID REFERENCES responses(id) ON DELETE CASCADE,
    score DECIMAL(5,2),
    feedback TEXT,
    evaluation_details JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'COMPLETED',
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    dataset_version VARCHAR(100),
    training_run VARCHAR(150),
    inference_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE evaluations ENABLE ROW LEVEL SECURITY;

-- 6. SKILL_EVIDENCE
CREATE TABLE IF NOT EXISTS skill_evidence (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    evaluation_id UUID REFERENCES evaluations(id) ON DELETE CASCADE,
    skill_name VARCHAR(150) NOT NULL,
    signal_strength DECIMAL(5,2),
    evidence_text TEXT,
    confidence VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE skill_evidence ENABLE ROW LEVEL SECURITY;

-- Normalize existing data from interview_questions to responses and evaluations (if legacy columns exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'interview_questions' AND column_name = 'answer_text') THEN
        INSERT INTO responses (id, question_id, response_text, submitted_at, created_at, updated_at)
        SELECT 
            uuid_generate_v4(), 
            id, 
            answer_text, 
            updated_at, 
            created_at, 
            updated_at
        FROM interview_questions 
        WHERE answer_text IS NOT NULL;
        
        INSERT INTO evaluations (response_id, score, feedback, evaluation_details)
        SELECT 
            r.id, 
            q.evaluation_score, 
            q.evaluation_feedback, 
            q.evaluation_details
        FROM interview_questions q
        JOIN responses r ON r.question_id = q.id
        WHERE q.evaluation_score IS NOT NULL OR q.evaluation_feedback IS NOT NULL;
        
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS answer_text;
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS evaluation_score;
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS evaluation_feedback;
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS evaluation_details;
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS code_text;
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS programming_language;
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS code_evaluation_score;
        ALTER TABLE interview_questions DROP COLUMN IF EXISTS code_evaluation_details;
    END IF;
END $$;

-- 7. CANDIDATE_SKILL_PROFILES
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'skill_profiles') AND 
       NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'candidate_skill_profiles') THEN
        EXECUTE 'ALTER TABLE public.skill_profiles RENAME TO candidate_skill_profiles';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS candidate_skill_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    skill_name VARCHAR(150) NOT NULL,
    estimated_proficiency DECIMAL(5,2) DEFAULT 0.00,
    confidence VARCHAR(50) DEFAULT 'insufficient evidence',
    evidence_count INTEGER DEFAULT 0,
    recent_performance DECIMAL(5,2) DEFAULT 0.00,
    historical_performance DECIMAL(5,2) DEFAULT 0.00,
    improvement_trend VARCHAR(50) DEFAULT 'neutral',
    last_evaluated_timestamp TIMESTAMP WITH TIME ZONE,
    evidence_history JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE candidate_skill_profiles ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_candidate_skill_profiles_user_id ON candidate_skill_profiles(user_id);

-- 8. RECOMMENDATIONS
CREATE TABLE IF NOT EXISTS recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES interview_sessions(id) ON DELETE SET NULL,
    target_skill VARCHAR(150) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    reason TEXT NOT NULL,
    evidence JSONB DEFAULT '{}',
    recommended_activity JSONB DEFAULT '{}',
    priority VARCHAR(20) NOT NULL,
    expected_learning_objective TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    baseline_proficiency DECIMAL(5,2) DEFAULT 0.00,
    post_outcome_proficiency DECIMAL(5,2),
    outcome_delta DECIMAL(5,2),
    outcome_assessment VARCHAR(50)
);
ALTER TABLE recommendations ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id);

-- 9. PRACTICE_SESSIONS
CREATE TABLE IF NOT EXISTS practice_sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    interview_type VARCHAR(100) NOT NULL,
    topic VARCHAR(100) NOT NULL,
    difficulty VARCHAR(20) DEFAULT 'medium',
    questions_attempted INTEGER DEFAULT 0,
    questions_correct INTEGER DEFAULT 0,
    total_score DECIMAL(5,2) DEFAULT 0.00,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE practice_sessions ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_practice_sessions_user_id ON practice_sessions(user_id);

-- 10. CODE_SUBMISSIONS
CREATE TABLE IF NOT EXISTS code_submissions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    question_id UUID REFERENCES interview_questions(id) ON DELETE CASCADE,
    code_text TEXT NOT NULL,
    programming_language VARCHAR(50) NOT NULL,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE code_submissions ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_code_submissions_question_id ON code_submissions(question_id);

-- 11. EXECUTION_RUNS
CREATE TABLE IF NOT EXISTS execution_runs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    code_submission_id UUID REFERENCES code_submissions(id) ON DELETE CASCADE,
    execution_time_ms INTEGER,
    memory_usage_kb INTEGER,
    stdout TEXT,
    stderr TEXT,
    exit_code INTEGER,
    test_cases_passed INTEGER DEFAULT 0,
    total_test_cases INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'COMPLETED',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE execution_runs ENABLE ROW LEVEL SECURITY;

-- 12. SESSION_EVENTS
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'logs') AND 
       NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'session_events') THEN
        EXECUTE 'ALTER TABLE public.logs RENAME TO session_events';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS session_events (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE session_events ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON session_events(session_id);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'session_events' AND column_name = 'details') THEN
        EXECUTE 'ALTER TABLE public.session_events RENAME COLUMN details TO event_data';
    END IF;
END $$;

-- 13. INTEGRITY_EVENTS
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'security_events') AND 
       NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'integrity_events') THEN
        EXECUTE 'ALTER TABLE public.security_events RENAME TO integrity_events';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS integrity_events (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    confidence DECIMAL(5,2) DEFAULT 0.00,
    evidence_details JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE integrity_events ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_integrity_events_session_id ON integrity_events(session_id);

DO $$
BEGIN
    -- Merge anomalies into integrity_events if table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'anomalies') THEN
        EXECUTE 'INSERT INTO integrity_events (session_id, event_type, severity, evidence_details, timestamp, created_at)
        SELECT session_id, anomaly_type, severity, details, timestamp, created_at
        FROM public.anomalies';
        EXECUTE 'DROP TABLE public.anomalies';
    END IF;
    
    -- Rename columns if migrating from security_events
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'integrity_events' AND column_name = 'security_level') THEN
        EXECUTE 'ALTER TABLE public.integrity_events RENAME COLUMN security_level TO severity';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'integrity_events' AND column_name = 'confidence_score') THEN
        EXECUTE 'ALTER TABLE public.integrity_events RENAME COLUMN confidence_score TO confidence';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'integrity_events' AND column_name = 'details') THEN
        EXECUTE 'ALTER TABLE public.integrity_events RENAME COLUMN details TO evidence_details';
    END IF;
END $$;

-- 14. AI_RUNS
CREATE TABLE IF NOT EXISTS ai_runs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
    prompt_name VARCHAR(100),
    model_version VARCHAR(100),
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    status VARCHAR(20) DEFAULT 'COMPLETED',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE ai_runs ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_ai_runs_session_id ON ai_runs(session_id);

-- Update trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_responses_updated_at ON responses;
CREATE TRIGGER update_responses_updated_at BEFORE UPDATE ON responses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_evaluations_updated_at ON evaluations;
CREATE TRIGGER update_evaluations_updated_at BEFORE UPDATE ON evaluations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Clean up obsolete columns from interview_sessions
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'interview_sessions' AND column_name = 'security_score') THEN
        EXECUTE 'ALTER TABLE public.interview_sessions DROP COLUMN IF EXISTS security_score';
        EXECUTE 'ALTER TABLE public.interview_sessions DROP COLUMN IF EXISTS security_events_count';
        EXECUTE 'ALTER TABLE public.interview_sessions DROP COLUMN IF EXISTS last_security_check';
    END IF;
END $$;

-- 15. QUESTION_SKILL_PREDICTIONS (ML-derived skill metadata)
CREATE TABLE IF NOT EXISTS question_skill_predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    question_id UUID REFERENCES interview_questions(id) ON DELETE CASCADE,
    predicted_skills JSONB NOT NULL,
    confidence DECIMAL(5,4) DEFAULT 0.0000,
    model_name VARCHAR(100) DEFAULT 'question-skill-v1',
    model_version VARCHAR(100) NOT NULL,
    dataset_version VARCHAR(100) DEFAULT '2026.10',
    training_run VARCHAR(150) DEFAULT 'question-skill-tagger-v1_20261003T063901Z',
    inference_timestamp TIMESTAMP WITH TIME ZONE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE question_skill_predictions ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_question_skill_predictions_question_id ON question_skill_predictions(question_id);

-- 16. MODEL_INFERENCES (Immutable model inference lineage across platform)
CREATE TABLE IF NOT EXISTS model_inferences (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id UUID REFERENCES interview_sessions(id) ON DELETE SET NULL,
    question_id UUID REFERENCES interview_questions(id) ON DELETE SET NULL,
    response_id UUID REFERENCES responses(id) ON DELETE SET NULL,
    evaluation_id UUID REFERENCES evaluations(id) ON DELETE SET NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    dataset_version VARCHAR(100) NOT NULL,
    training_run VARCHAR(150) NOT NULL,
    inference_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    inference_output JSONB DEFAULT '{}',
    source VARCHAR(50) DEFAULT 'ml',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
ALTER TABLE model_inferences ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_model_inferences_session_id ON model_inferences(session_id);
CREATE INDEX IF NOT EXISTS idx_model_inferences_model_name ON model_inferences(model_name);
CREATE INDEX IF NOT EXISTS idx_model_inferences_version ON model_inferences(model_name, model_version);

-- 17. PERMISSIONS & RLS POLICIES FOR PLATFORM ACCESS
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL ROUTINES IN SCHEMA public TO anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO anon, authenticated;

-- Configure RLS policies for all tables
DO $$
DECLARE
    tbl text;
    tables text[] := ARRAY[
        'profiles',
        'interview_sessions',
        'interview_questions',
        'responses',
        'evaluations',
        'skill_evidence',
        'candidate_skill_profiles',
        'recommendations',
        'practice_sessions',
        'code_submissions',
        'execution_runs',
        'session_events',
        'integrity_events',
        'ai_runs',
        'question_skill_predictions',
        'model_inferences'
    ];
BEGIN
    FOREACH tbl IN ARRAY tables LOOP
        EXECUTE format('DROP POLICY IF EXISTS "allow_all_authenticated_%s" ON %I', tbl, tbl);
        EXECUTE format('CREATE POLICY "allow_all_authenticated_%s" ON %I FOR ALL TO authenticated USING (true) WITH CHECK (true)', tbl, tbl);
        EXECUTE format('DROP POLICY IF EXISTS "allow_all_anon_%s" ON %I', tbl, tbl);
        EXECUTE format('CREATE POLICY "allow_all_anon_%s" ON %I FOR ALL TO anon USING (true) WITH CHECK (true)', tbl, tbl);
    END LOOP;
END $$;
