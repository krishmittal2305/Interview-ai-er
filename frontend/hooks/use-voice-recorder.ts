"use client"

/**
 * useVoiceRecorder — push-to-talk audio capture hook with full state management.
 *
 * Features:
 *   - Push-to-talk (hold) or toggle (click) recording modes
 *   - Audio permission request + error handling
 *   - Clear recording state machine: idle → requesting → recording → processing → done / error
 *   - Sends audio to /api/transcribe, returns transcript + metadata
 *   - Transcript editing before submission
 *   - Latency tracking at every step
 *   - Failure recovery (retry mechanism)
 *   - Does NOT persist raw audio
 */

import { useState, useRef, useCallback, useEffect } from "react"
import { API_BASE_URL as API_BASE } from "@/lib/api-client"

// ─────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────

export type RecordingState =
  | "idle"
  | "requesting_permission"
  | "recording"
  | "processing"
  | "transcribing"
  | "done"
  | "error"

export interface TranscriptionMetadata {
  transcript: string
  language: string
  duration_seconds: number
  model_id: string
  confidence: number | null
  latency_ms: number
  timestamp: string
  segments: Array<{ text: string; start: number | null; end: number | null }>
}

export interface VoiceRecorderState {
  /** Current state of the recording pipeline */
  recordingState: RecordingState
  /** Whether the microphone is currently capturing */
  isRecording: boolean
  /** Whether the transcription is being processed */
  isTranscribing: boolean
  /** The transcript text (editable) */
  transcript: string
  /** Full transcription metadata from Whisper */
  transcriptionMetadata: TranscriptionMetadata | null
  /** Error message if something went wrong */
  errorMessage: string | null
  /** Duration of the current/last recording in seconds */
  recordingDuration: number
  /** Latency breakdown for observability */
  latency: {
    permissionMs: number | null
    recordingMs: number | null
    uploadMs: number | null
    transcriptionMs: number | null
    totalMs: number | null
  }
  /** Whether the browser supports audio recording */
  isSupported: boolean
  /** Whether microphone permission has been granted */
  hasPermission: boolean | null
}

export interface VoiceRecorderActions {
  /** Start recording audio */
  startRecording: () => Promise<void>
  /** Stop recording and begin transcription */
  stopRecording: () => Promise<void>
  /** Toggle recording on/off */
  toggleRecording: () => Promise<void>
  /** Update the transcript text (for editing before submission) */
  setTranscript: (text: string) => void
  /** Reset the recorder to idle state */
  reset: () => void
  /** Retry the last failed transcription */
  retry: () => Promise<void>
}

// ─────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────


export function useVoiceRecorder(): [VoiceRecorderState, VoiceRecorderActions] {
  // State
  const [recordingState, setRecordingState] = useState<RecordingState>("idle")
  const [transcript, setTranscript] = useState("")
  const [transcriptionMetadata, setTranscriptionMetadata] = useState<TranscriptionMetadata | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)
  const [latency, setLatency] = useState<VoiceRecorderState["latency"]>({
    permissionMs: null,
    recordingMs: null,
    uploadMs: null,
    transcriptionMs: null,
    totalMs: null,
  })

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const recordingStartRef = useRef<number>(0)
  const pipelineStartRef = useRef<number>(0)
  const durationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastAudioBlobRef = useRef<Blob | null>(null)

  // Browser support check
  const isSupported = typeof window !== "undefined" && !!navigator?.mediaDevices?.getUserMedia

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (durationIntervalRef.current) clearInterval(durationIntervalRef.current)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  // ── Permission ─────────────────────────────────────────────

  const requestPermission = useCallback(async (): Promise<MediaStream> => {
    setRecordingState("requesting_permission")
    const t0 = performance.now()

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      })
      setHasPermission(true)
      setLatency((prev) => ({ ...prev, permissionMs: Math.round(performance.now() - t0) }))
      return stream
    } catch (err: any) {
      setHasPermission(false)
      const msg =
        err.name === "NotAllowedError"
          ? "Microphone permission was denied. Please allow microphone access in your browser settings."
          : err.name === "NotFoundError"
          ? "No microphone detected. Please connect a microphone and try again."
          : `Microphone error: ${err.message}`
      throw new Error(msg)
    }
  }, [])

  // ── Start recording ────────────────────────────────────────

  const startRecording = useCallback(async () => {
    if (!isSupported) {
      setErrorMessage("Audio recording is not supported in this browser.")
      setRecordingState("error")
      return
    }

    setErrorMessage(null)
    pipelineStartRef.current = performance.now()

    try {
      const stream = await requestPermission()
      streamRef.current = stream
      chunksRef.current = []

      // Choose the best supported MIME type
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/wav"

      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      recorder.start(250) // Collect data every 250ms
      recordingStartRef.current = performance.now()
      setRecordingState("recording")
      setRecordingDuration(0)

      // Live duration counter
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(
          Math.round((performance.now() - recordingStartRef.current) / 1000)
        )
      }, 200)
    } catch (err: any) {
      setErrorMessage(err.message)
      setRecordingState("error")
    }
  }, [isSupported, requestPermission])

  // ── Stop recording ─────────────────────────────────────────

  const stopRecording = useCallback(async () => {
    const recorder = mediaRecorderRef.current
    if (!recorder || recorder.state === "inactive") return

    // Stop the duration counter
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current)
      durationIntervalRef.current = null
    }

    const recordingMs = Math.round(performance.now() - recordingStartRef.current)
    setLatency((prev) => ({ ...prev, recordingMs }))

    // Wait for the recorder to finish
    const audioBlob = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || "audio/webm"
        const blob = new Blob(chunksRef.current, { type: mimeType })
        resolve(blob)
      }
      recorder.stop()
    })

    // Release the stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }

    lastAudioBlobRef.current = audioBlob
    await sendForTranscription(audioBlob)
  }, [])

  // ── Send to backend ────────────────────────────────────────

  const sendForTranscription = useCallback(async (blob: Blob) => {
    setRecordingState("transcribing")

    const uploadStart = performance.now()

    try {
      const formData = new FormData()
      formData.append("file", blob, "recording.webm")

      // Get auth token
      let authHeader: string | undefined
      try {
        const { getBrowserSupabaseClient } = await import("@/lib/supabase")
        const supabase = getBrowserSupabaseClient()
        if (supabase) {
          const { data } = await supabase.auth.getSession()
          if (data.session?.access_token) {
            authHeader = `Bearer ${data.session.access_token}`
          }
        }
      } catch {
        // Auth token optional in dev
      }

      const headers: Record<string, string> = {}
      if (authHeader) headers["Authorization"] = authHeader

      const response = await fetch(`${API_BASE}/transcribe`, {
        method: "POST",
        headers,
        body: formData,
      })

      const uploadMs = Math.round(performance.now() - uploadStart)

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.message || `Transcription failed: HTTP ${response.status}`)
      }

      const json = await response.json()
      const data = json.data || json

      const totalMs = Math.round(performance.now() - pipelineStartRef.current)

      setTranscript(data.transcript || "")
      setTranscriptionMetadata({
        transcript: data.transcript || "",
        language: data.language || "en",
        duration_seconds: data.duration_seconds || 0,
        model_id: data.model_id || "openai/whisper-base.en",
        confidence: data.confidence ?? null,
        latency_ms: data.latency_ms || 0,
        timestamp: data.timestamp || new Date().toISOString(),
        segments: data.segments || [],
      })
      setLatency((prev) => ({
        ...prev,
        uploadMs,
        transcriptionMs: data.latency_ms || null,
        totalMs,
      }))
      setRecordingState("done")
    } catch (err: any) {
      setErrorMessage(err.message)
      setRecordingState("error")
    }
  }, [])

  // ── Toggle ─────────────────────────────────────────────────

  const toggleRecording = useCallback(async () => {
    if (recordingState === "recording") {
      await stopRecording()
    } else if (recordingState === "idle" || recordingState === "done" || recordingState === "error") {
      await startRecording()
    }
  }, [recordingState, startRecording, stopRecording])

  // ── Retry ──────────────────────────────────────────────────

  const retry = useCallback(async () => {
    if (lastAudioBlobRef.current) {
      setErrorMessage(null)
      await sendForTranscription(lastAudioBlobRef.current)
    } else {
      // No audio to retry — start a fresh recording
      await startRecording()
    }
  }, [sendForTranscription, startRecording])

  // ── Reset ──────────────────────────────────────────────────

  const reset = useCallback(() => {
    setRecordingState("idle")
    setTranscript("")
    setTranscriptionMetadata(null)
    setErrorMessage(null)
    setRecordingDuration(0)
    setLatency({
      permissionMs: null,
      recordingMs: null,
      uploadMs: null,
      transcriptionMs: null,
      totalMs: null,
    })
    lastAudioBlobRef.current = null
    chunksRef.current = []
  }, [])

  // ── Return ─────────────────────────────────────────────────

  const state: VoiceRecorderState = {
    recordingState,
    isRecording: recordingState === "recording",
    isTranscribing: recordingState === "transcribing",
    transcript,
    transcriptionMetadata,
    errorMessage,
    recordingDuration,
    latency,
    isSupported,
    hasPermission,
  }

  const actions: VoiceRecorderActions = {
    startRecording,
    stopRecording,
    toggleRecording,
    setTranscript,
    reset,
    retry,
  }

  return [state, actions]
}
