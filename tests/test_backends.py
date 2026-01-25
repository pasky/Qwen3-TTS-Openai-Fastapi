# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for backend selection and initialization.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.backends.factory import get_backend, reset_backend
from api.backends.base import TTSBackend
from api.backends.official_qwen3_tts import OfficialQwen3TTSBackend
from api.backends.vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend


class TestBackendSelection:
    """Test backend selection via environment variables."""
    
    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()
    
    def test_default_backend_is_official(self, monkeypatch):
        """Test that official backend is selected by default."""
        # Ensure TTS_BACKEND is not set
        monkeypatch.delenv("TTS_BACKEND", raising=False)
        
        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"
    
    def test_official_backend_via_env(self, monkeypatch):
        """Test selecting official backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        
        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"
    
    def test_vllm_backend_via_env(self, monkeypatch):
        """Test selecting vLLM-Omni backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "vllm_omni")
        
        backend = get_backend()
        assert isinstance(backend, VLLMOmniQwen3TTSBackend)
        assert backend.get_backend_name() == "vllm_omni"
    
    def test_vllm_backend_alternate_name(self, monkeypatch):
        """Test vLLM backend with alternate name format."""
        monkeypatch.setenv("TTS_BACKEND", "vllm-omni")
        
        backend = get_backend()
        assert isinstance(backend, VLLMOmniQwen3TTSBackend)
        assert backend.get_backend_name() == "vllm_omni"
    
    def test_invalid_backend_raises_error(self, monkeypatch):
        """Test that invalid backend name raises ValueError."""
        monkeypatch.setenv("TTS_BACKEND", "invalid_backend")
        
        with pytest.raises(ValueError, match="Unknown TTS_BACKEND"):
            get_backend()
    
    def test_custom_model_name_via_env(self, monkeypatch):
        """Test overriding model name via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        monkeypatch.setenv("TTS_MODEL_NAME", "custom/model")
        
        backend = get_backend()
        assert backend.get_model_id() == "custom/model"
    
    def test_backend_singleton(self, monkeypatch):
        """Test that get_backend returns the same instance."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        
        backend1 = get_backend()
        backend2 = get_backend()
        
        assert backend1 is backend2


class TestBackendInterface:
    """Test that all backends implement the required interface."""
    
    def test_official_backend_implements_interface(self):
        """Test official backend implements TTSBackend interface."""
        backend = OfficialQwen3TTSBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
    
    def test_vllm_backend_implements_interface(self):
        """Test vLLM backend implements TTSBackend interface."""
        backend = VLLMOmniQwen3TTSBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
    
    def test_backend_names_are_correct(self):
        """Test that backends return correct names."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert official.get_backend_name() == "official"
        assert vllm.get_backend_name() == "vllm_omni"
    
    def test_backends_return_voices(self):
        """Test that backends return voice lists."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        # Both backends should return a list of voices
        assert isinstance(official.get_supported_voices(), list)
        assert isinstance(vllm.get_supported_voices(), list)
        assert len(official.get_supported_voices()) > 0
        assert len(vllm.get_supported_voices()) > 0
    
    def test_backends_return_languages(self):
        """Test that backends return language lists."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        # Both backends should return a list of languages
        assert isinstance(official.get_supported_languages(), list)
        assert isinstance(vllm.get_supported_languages(), list)
        assert len(official.get_supported_languages()) > 0
        assert len(vllm.get_supported_languages()) > 0
    
    def test_backends_initially_not_ready(self):
        """Test that backends are not ready before initialization."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert not official.is_ready()
        assert not vllm.is_ready()
    
    def test_backends_return_device_info(self):
        """Test that backends return device info dict."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        info1 = official.get_device_info()
        info2 = vllm.get_device_info()
        
        # Check required keys
        assert "device" in info1
        assert "gpu_available" in info1
        assert "device" in info2
        assert "gpu_available" in info2
