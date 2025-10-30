import asyncio
import re
import threading
import numpy as np
import logging
import time
import tempfile
import soundfile as sf
from typing import List
from pathlib import Path

from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from whisperlivekit.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)

def extract_number(s: str) -> int:
    """Extract number from speaker label (e.g., 'SPEAKER_00' -> 0)"""
    m = re.search(r'\d+', s)
    return int(m.group()) if m else 0


class DiarizationProcessor:
    """Processes audio chunks for diarization using pyannote.audio pipeline."""
    
    def __init__(self, pipeline: SpeakerDiarization, sample_rate: int = 16000, processing_interval: float = 2.0):
        """
        Initialize the diarization processor.
        
        Args:
            pipeline: pyannote.audio SpeakerDiarization pipeline
            sample_rate: Audio sample rate in Hz
            processing_interval: How often to run diarization (in seconds)
        """
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        self.processing_interval = processing_interval
        
        self.speaker_segments = []
        self.segment_lock = threading.Lock()
        self.global_time_offset = 0.0
        
        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.total_audio_duration = 0.0
        self.last_processed_duration = 0.0
        
        # Processing control
        self._running = False
        self._processing_thread = None
        
    def start(self):
        """Start the background processing thread."""
        if not self._running:
            self._running = True
            self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
            self._processing_thread.start()
            logger.info("Diarization processor started")
    
    def stop(self):
        """Stop the background processing thread."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
        logger.info("Diarization processor stopped")
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to the buffer."""
        with self.buffer_lock:
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            self.total_audio_duration = len(self.audio_buffer) / self.sample_rate
    
    def _process_loop(self):
        """Background processing loop."""
        while self._running:
            time.sleep(self.processing_interval)
            
            with self.buffer_lock:
                # Check if we have enough new audio to process
                new_audio_duration = self.total_audio_duration - self.last_processed_duration
                
                if new_audio_duration < self.processing_interval * 0.5:
                    continue
                
                # Get audio to process
                audio_to_process = self.audio_buffer.copy()
                processing_duration = self.total_audio_duration
            
            if len(audio_to_process) == 0:
                continue
                
            try:
                # Run diarization on accumulated audio
                segments = self._run_diarization(audio_to_process, processing_duration)
                
                with self.segment_lock:
                    # Update segments - only keep new segments beyond what we've processed
                    self.speaker_segments = [
                        seg for seg in self.speaker_segments 
                        if seg.start < self.last_processed_duration + self.global_time_offset
                    ] + segments
                
                self.last_processed_duration = processing_duration
                
            except Exception as e:
                logger.error(f"Error in diarization processing: {e}")
    
    def _run_diarization(self, audio: np.ndarray, duration: float) -> List[SpeakerSegment]:
        """
        Run diarization on audio buffer.
        
        Args:
            audio: Audio data as numpy array
            duration: Duration of audio in seconds
            
        Returns:
            List of SpeakerSegment objects
        """
        segments = []
        
        try:
            # Create a temporary WAV file for pyannote
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio, self.sample_rate)
            
            # Run diarization
            diarization = self.pipeline(tmp_path)
            
            # Convert pyannote output to SpeakerSegment objects
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Only include segments from new audio
                if turn.start >= self.last_processed_duration:
                    segments.append(SpeakerSegment(
                        speaker=speaker,
                        start=turn.start + self.global_time_offset,
                        end=turn.end + self.global_time_offset
                    ))
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            logger.debug(f"Diarization found {len(segments)} new segments")
            
        except Exception as e:
            logger.error(f"Error running diarization: {e}")
        
        return segments
    
    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.speaker_segments.copy()
    
    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.total_audio_duration + self.global_time_offset
            self.speaker_segments = [
                segment for segment in self.speaker_segments 
                if current_time - segment.end < older_than
            ]
    
    def insert_silence(self, silence_duration: float):
        """Adjust global time offset for silence periods."""
        self.global_time_offset += silence_duration
        logger.debug(f"Inserted silence of {silence_duration:.2f}s, new offset: {self.global_time_offset:.2f}s")




class DiartDiarization:
    """
    Diarization backend using pyannote.audio SpeakerDiarization pipeline.
    
    This class maintains compatibility with the original diart-based interface
    while using the newer pyannote.audio API underneath.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        config=None,  # Kept for compatibility but not used
        use_microphone: bool = False,  # Not supported with pyannote approach
        block_duration: float = 1.5,
        segmentation_model_name: str = "pyannote/segmentation-3.0",
        embedding_model_name: str = "pyannote/embedding",
        pipeline_name: str = "pyannote/speaker-diarization-3.1"
    ):
        """
        Initialize diarization using pyannote.audio pipeline.
        
        Args:
            sample_rate: Audio sample rate (should be 16000)
            config: Ignored, kept for compatibility
            use_microphone: Not supported with pyannote approach
            block_duration: How often to process accumulated audio
            segmentation_model_name: Ignored (pipeline uses its own)
            embedding_model_name: Ignored (pipeline uses its own)
            pipeline_name: HuggingFace model ID for the diarization pipeline
        """
        if use_microphone:
            logger.warning("Microphone input not supported with pyannote backend, using buffer mode")
        
        self.sample_rate = sample_rate
        self.lag_diart = None  # Kept for compatibility with assign_speakers_to_tokens
        
        # Load the pyannote.audio pipeline
        logger.info(f"Loading pyannote.audio pipeline: {pipeline_name}")
        try:
            self.pipeline = SpeakerDiarization.from_pretrained(pipeline_name)
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            logger.error("Make sure you have accepted the user conditions for pyannote models:")
            logger.error("1. https://huggingface.co/pyannote/segmentation-3.0")
            logger.error("2. https://huggingface.co/pyannote/speaker-diarization-3.1")
            logger.error("And run: huggingface-cli login")
            raise
        
        # Create processor
        self.processor = DiarizationProcessor(
            pipeline=self.pipeline,
            sample_rate=sample_rate,
            processing_interval=block_duration
        )
        
        # Start processing
        self.processor.start()
        
        logger.info("DiartDiarization initialized with pyannote.audio backend")
    
    def insert_silence(self, silence_duration: float):
        """Insert silence period by adjusting the global time offset."""
        self.processor.insert_silence(silence_duration)
    
    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization.
        
        Args:
            pcm_array: Audio data as numpy array
        """
        self.processor.add_audio(pcm_array)
    
    def close(self):
        """Stop the diarization processor."""
        self.processor.stop()
    
    def assign_speakers_to_tokens(self, tokens: list, use_punctuation_split: bool = False) -> list:
        """
        Assign speakers to tokens based on timing overlap with speaker segments.
        
        Args:
            tokens: List of tokens with timing information
            use_punctuation_split: If True, uses punctuation marks to refine speaker boundaries
            
        Returns:
            List of tokens with speaker assignments
        """
        segments = self.processor.get_segments()
        
        # Debug logging
        logger.debug(f"assign_speakers_to_tokens called with {len(tokens)} tokens")
        logger.debug(f"Available segments: {len(segments)}")
        for i, seg in enumerate(segments[:5]):  # Show first 5 segments
            logger.debug(f"  Segment {i}: {seg.speaker} [{seg.start:.2f}-{seg.end:.2f}]")
        
        if not self.lag_diart and segments and tokens:
            self.lag_diart = segments[0].start - tokens[0].start
        
        if not use_punctuation_split:
            for token in tokens:
                for segment in segments:
                    lag = self.lag_diart if self.lag_diart else 0
                    if not (segment.end <= token.start + lag or segment.start >= token.end + lag):
                        # Extract speaker number from label (e.g., "SPEAKER_00" -> 0)
                        speaker_num = extract_number(segment.speaker)
                        token.speaker = speaker_num + 1
        else:
            tokens = add_speaker_to_tokens(segments, tokens)
        
        return tokens

        
def concatenate_speakers(segments):
    segments_concatenated = [{"speaker": 1, "begin": 0.0, "end": 0.0}]
    for segment in segments:
        speaker = extract_number(segment.speaker) + 1
        if segments_concatenated[-1]['speaker'] != speaker:
            segments_concatenated.append({"speaker": speaker, "begin": segment.start, "end": segment.end})
        else:
            segments_concatenated[-1]['end'] = segment.end
    # print("Segments concatenated:")
    # for entry in segments_concatenated:
    #     print(f"Speaker {entry['speaker']}: {entry['begin']:.2f}s - {entry['end']:.2f}s")   
    return segments_concatenated


def add_speaker_to_tokens(segments, tokens):
    """
    Assign speakers to tokens based on diarization segments, with punctuation-aware boundary adjustment.
    """
    punctuation_marks = {'.', '!', '?'}
    punctuation_tokens = [token for token in tokens if token.text.strip() in punctuation_marks]
    segments_concatenated = concatenate_speakers(segments)
    for ind, segment in enumerate(segments_concatenated):
            for i, punctuation_token in enumerate(punctuation_tokens):
                if punctuation_token.start > segment['end']:
                    after_length = punctuation_token.start - segment['end']
                    before_length = segment['end'] - punctuation_tokens[i - 1].end
                    if before_length > after_length:
                        segment['end'] = punctuation_token.start
                        if i < len(punctuation_tokens) - 1 and ind + 1 < len(segments_concatenated):
                            segments_concatenated[ind + 1]['begin'] = punctuation_token.start
                    else:
                        segment['end'] = punctuation_tokens[i - 1].end
                        if i < len(punctuation_tokens) - 1 and ind - 1 >= 0:
                            segments_concatenated[ind - 1]['begin'] = punctuation_tokens[i - 1].end
                    break

    last_end = 0.0
    for token in tokens:
        start = max(last_end + 0.01, token.start)
        token.start = start
        token.end = max(start, token.end)
        last_end = token.end

    ind_last_speaker = 0
    for segment in segments_concatenated:
        for i, token in enumerate(tokens[ind_last_speaker:]):
            if token.end <= segment['end']:
                token.speaker = segment['speaker']
                ind_last_speaker = i + 1
                # print(
                #     f"Token '{token.text}' ('begin': {token.start:.2f}, 'end': {token.end:.2f}) "
                #     f"assigned to Speaker {segment['speaker']} ('segment': {segment['begin']:.2f}-{segment['end']:.2f})"
                # )
            elif token.start > segment['end']:
                break
    return tokens


def visualize_tokens(tokens):
    conversation = [{"speaker": -1, "text": ""}]
    for token in tokens:
        speaker = conversation[-1]['speaker']
        if token.speaker != speaker:
            conversation.append({"speaker": token.speaker, "text": token.text})
        else:
            conversation[-1]['text'] += token.text
    print("Conversation:")
    for entry in conversation:
        print(f"Speaker {entry['speaker']}: {entry['text']}")