"""Pydantic models for request/response validation."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class VADParameters(BaseModel):
    """Voice Activity Detection parameters."""

    prob_threshold: Optional[float] = Field(
        None,
        ge=0.1,
        le=0.9,
        description="Speech probability threshold (0.1-0.9)"
    )
    silence_duration: Optional[float] = Field(
        None,
        ge=0.1,
        le=5.0,
        description="Silence duration to end utterance (0.1-5 seconds)"
    )
    min_speech_duration: Optional[float] = Field(
        None,
        ge=0.1,
        le=2.0,
        description="Minimum speech duration to transcribe (0.1-2 seconds)"
    )
    sample_rate: Optional[int] = Field(
        None,
        ge=8000,
        le=48000,
        description="Sample rate for VAD (8000-48000 Hz)"
    )
    window_size: Optional[int] = Field(
        None,
        ge=256,
        le=2048,
        description="VAD window size (256-2048 samples)"
    )


class AudioParameters(BaseModel):
    """Audio processing parameters."""

    channels: Optional[int] = Field(
        None,
        ge=1,
        le=2,
        description="Number of audio channels (1-2)"
    )
    sample_rate: Optional[int] = Field(
        None,
        ge=8000,
        le=48000,
        description="Audio sample rate (8000-48000 Hz)"
    )
    sample_width: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description="Sample width in bytes (1-4)"
    )


class TranscriptionParameters(BaseModel):
    """Transcription parameters."""

    context_max_length: Optional[int] = Field(
        None,
        ge=0,
        le=500,
        description="Maximum context length for Whisper (0-500 chars)"
    )


class CleanupParameters(BaseModel):
    """Resource cleanup parameters."""

    session_ttl_minutes: Optional[int] = Field(
        None,
        ge=1,
        le=1440,
        description="Session TTL in minutes (1-1440)"
    )
    job_retention_minutes: Optional[int] = Field(
        None,
        ge=1,
        le=1440,
        description="Job retention time in minutes (1-1440)"
    )
    cleanup_interval_seconds: Optional[int] = Field(
        None,
        ge=60,
        le=3600,
        description="Cleanup interval in seconds (60-3600)"
    )


class UploadParameters(BaseModel):
    """File upload parameters."""

    max_file_size_mb: Optional[int] = Field(
        None,
        ge=1,
        le=500,
        description="Maximum file size in MB (1-500)"
    )


class RateLimitingParameters(BaseModel):
    """Rate limiting parameters."""

    enabled: Optional[bool] = Field(
        None,
        description="Whether rate limiting is enabled"
    )
    requests_per_minute: Optional[int] = Field(
        None,
        ge=1,
        le=1000,
        description="Maximum API requests per minute"
    )
    uploads_per_minute: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Maximum file uploads per minute"
    )


class PreloadModelsParameters(BaseModel):
    """Model preloading parameters."""

    enabled: Optional[bool] = Field(
        None,
        description="Whether to preload models on startup"
    )
    models: Optional[list] = Field(
        None,
        description="List of model names to preload"
    )


class AuthParameters(BaseModel):
    """Authentication parameters."""

    enabled: Optional[bool] = Field(
        None,
        description="Whether authentication is enabled"
    )
    api_key: Optional[str] = Field(
        None,
        description="API key for authentication"
    )


class SettingsUpdate(BaseModel):
    """Request model for updating application settings.

    All fields are optional to allow partial updates.
    """

    model_config = ConfigDict(extra="forbid")

    vad_parameters: Optional[VADParameters] = None
    audio_parameters: Optional[AudioParameters] = None
    transcription_parameters: Optional[TranscriptionParameters] = None
    cleanup_parameters: Optional[CleanupParameters] = None
    upload_parameters: Optional[UploadParameters] = None
    rate_limiting: Optional[RateLimitingParameters] = None
    preload_models: Optional[PreloadModelsParameters] = None
    auth: Optional[AuthParameters] = None
