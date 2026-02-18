"""Tests for the POST /v1/audio/transcriptions endpoint."""

import os

from utils.config import Settings

ENDPOINT = "/v1/audio/transcriptions"


def _post_to_transcribe_endpoint(client, audio_bytes, **form_fields):
    """Helper: POST a file upload to the transcription endpoint."""
    return client.post(
        ENDPOINT,
        files={"file": ("test.wav", audio_bytes, "audio/wav")},
        data=form_fields,
    )


# Test Success


class TestTranscribeSuccess:
    def test_default_params(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """Successful transcription with no explicit model or language."""
        response = _post_to_transcribe_endpoint(client, sample_audio_bytes)

        assert response.status_code == 200
        body = response.json()
        assert "segments" in body
        assert body["segments"][0]["words"][0]["word"] == "Hello"

    def test_with_language(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """Explicit language is forwarded to the transcribe service."""
        response = _post_to_transcribe_endpoint(
            client, sample_audio_bytes, language="en"
        )

        assert response.status_code == 200
        mock_transcribe.assert_called_once()
        assert mock_transcribe.call_args.args[2] == "en"

    def test_with_matching_model(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """Explicit model that matches the configured model succeeds."""
        response = _post_to_transcribe_endpoint(
            client, sample_audio_bytes, model="large-v2"
        )

        assert response.status_code == 200


# Test Validation Errors


class TestTranscribeValidation:
    def test_unsupported_transcribe_language(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """Language not in whisperx.utils.LANGUAGES returns 400."""
        response = _post_to_transcribe_endpoint(
            client, sample_audio_bytes, language="xx"
        )

        assert response.status_code == 400
        assert "Unsupported language" in response.json()["detail"]
        assert "for transcription" in response.json()["detail"]

    def test_unsupported_align_language(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """Language in LANGUAGES but missing from alignment dicts returns 400."""

        # NB: "cz" is supported for transcribe but not align
        response = _post_to_transcribe_endpoint(
            client, sample_audio_bytes, language="cz"
        )

        assert response.status_code == 400
        assert "Unsupported language" in response.json()["detail"]
        assert "for alignment" in response.json()["detail"]

    def test_wrong_model(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """Model that differs from configured model returns 404."""
        response = _post_to_transcribe_endpoint(
            client, sample_audio_bytes, model="tiny"
        )

        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]

    def test_missing_file(self, client, mock_whisperx, mock_transcribe):
        """Request without a file upload returns 422."""
        response = client.post(ENDPOINT)

        assert response.status_code == 422


# Behavior tests


class TestTranscribeBehaviour:
    def test_load_audio_called_with_temp_path(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """whisperx.load_audio is called with a temp file path that is cleaned up."""
        response = _post_to_transcribe_endpoint(client, sample_audio_bytes)

        assert response.status_code == 200
        mock_whisperx["load_audio"].assert_called_once()
        temp_path = mock_whisperx["load_audio"].call_args.args[0]
        assert isinstance(temp_path, str)
        # Temp file should have been deleted by the endpoint
        assert not os.path.exists(temp_path)

    def test_temp_file_extension_preserved(
        self, client, mock_whisperx, mock_transcribe
    ):
        """Temp file preserves the original upload extension."""
        response = client.post(
            ENDPOINT,
            files={"file": ("interview.ogg", b"\x00" * 512, "audio/mpeg")},
        )

        assert response.status_code == 200
        temp_path = mock_whisperx["load_audio"].call_args.args[0]
        assert temp_path.endswith(".ogg")

    def test_transcribe_called_with_correct_args(
        self, client, mock_whisperx, mock_transcribe, sample_audio_bytes
    ):
        """The transcribe service receives (audio_array, settings, language)."""
        response = _post_to_transcribe_endpoint(
            client, sample_audio_bytes, language="fr"
        )

        assert response.status_code == 200
        mock_transcribe.assert_called_once()
        args = mock_transcribe.call_args.args
        # 0: Numpy audio array returned by load_audio
        assert args[0] is mock_whisperx["fake_audio"]
        # 1: Settings instance with expected values
        assert isinstance(args[1], Settings)
        assert args[1].transcribe_model == "large-v2"
        assert args[1].batch_size == 4
        # 2: language
        assert args[2] == "fr"
