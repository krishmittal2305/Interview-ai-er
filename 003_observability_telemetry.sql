-- Migration: 003_observability_telemetry.sql
-- Table: ai_inference_logs
-- Observability table across both ML inference and LLM inference.
-- Stores model, task, latency, success/failure, input size, output size, memory/device, fallback usage, and retry count.

CREATE TABLE IF NOT EXISTS ai_inference_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    inference_type VARCHAR(16) NOT NULL, -- 'ml' or 'llm'
    model VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) DEFAULT 'v1.0',
    task VARCHAR(128) NOT NULL,
    latency_ms NUMERIC(10, 2) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT true,
    status VARCHAR(32) NOT NULL DEFAULT 'SUCCESS', -- 'SUCCESS', 'FAILED', 'FALLBACK', 'TIMEOUT'
    error_message TEXT,
    input_size INTEGER DEFAULT 0,
    output_size INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    memory_device VARCHAR(64) DEFAULT 'cpu',
    fallback_usage BOOLEAN NOT NULL DEFAULT false,
    retry_count INTEGER NOT NULL DEFAULT 0,
    cost_estimate_usd NUMERIC(10, 6) DEFAULT 0.000000,
    metadata JSONB DEFAULT '{}'
);

-- Indexing for observability metrics, time-series querying, and analytics
CREATE INDEX IF NOT EXISTS idx_ai_inference_logs_type ON ai_inference_logs(inference_type);
CREATE INDEX IF NOT EXISTS idx_ai_inference_logs_timestamp ON ai_inference_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_inference_logs_model ON ai_inference_logs(model);
CREATE INDEX IF NOT EXISTS idx_ai_inference_logs_task ON ai_inference_logs(task);
CREATE INDEX IF NOT EXISTS idx_ai_inference_logs_status ON ai_inference_logs(status);
CREATE INDEX IF NOT EXISTS idx_ai_inference_logs_latency ON ai_inference_logs(latency_ms DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE ai_inference_logs ENABLE ROW LEVEL SECURITY;

-- Allow authenticated and service role to read/write observability telemetry
CREATE POLICY "Allow service role full access to inference logs"
ON ai_inference_logs
FOR ALL
USING (true)
WITH CHECK (true);
