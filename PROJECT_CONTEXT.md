# Project Context: hypa-thymesia-video-query

## Overview

A semantic video search system that enables natural language querying of video content. Users can upload videos, index them by extracting frames and transcribing audio, then search using text queries to find matching visual frames and spoken words. Supports queries like "red balloon" (visual), "person speaking about AI" (transcript), or combined multimodal search.

## Core Technology

- **CLIP (openai/clip-vit-base-patch32)** - Generates 512-dimensional semantic embeddings for images
- **Faster-Whisper** - Optimized speech-to-text transcription (4x faster than openai-whisper, lower memory usage)
- **Sentence-Transformers (all-MiniLM-L6-v2)** - Text embedding model for transcript search
- **ChromaDB** - Vector database with cosine similarity search (separate collections for frames and transcripts)
- **OpenCV** - Video processing and frame extraction
- **Gradio** - Web UI for upload, indexing, and search
- **PyTorch** - Deep learning framework (GPU/CPU auto-detection)

## System Architecture

### Indexing Pipeline

**Visual Pipeline:**
1. **Frame Extraction** - Samples video at configurable intervals (default 1.5s)
2. **Solid Frame Filtering** - Optional removal of blank/static frames using edge detection and color analysis
3. **Scene Detection** - HSV histogram comparison to identify scene boundaries (configurable threshold)
4. **Frame Embedding** - CLIP processes each frame into normalized 512-dim vectors
5. **Storage** - Frame embeddings stored in ChromaDB "video_frames" collection with metadata (timestamp, scene_id, video_path)

**Audio Pipeline:**
1. **Audio Transcription** - Faster-Whisper transcribes audio with segment-level timestamps and VAD filtering
2. **Chunking** - Transcript split into 20-second segments for context balance
3. **Text Embedding** - Sentence-Transformers generates embeddings for each chunk
4. **Storage** - Transcript embeddings stored in ChromaDB "video_transcripts" collection with metadata (start_time, end_time, text)

### Search Pipeline

**Frame Search:**
1. **Query Embedding** - Text query converted to 512-dim CLIP embedding
2. **Vector Search** - ChromaDB retrieves top-k similar frames using cosine distance
3. **Diversity Re-ranking** - Greedy algorithm balances similarity with scene/temporal diversity
4. **Result Display** - Frames reconstructed and displayed with similarity scores

**Transcript Search:**
1. **Query Embedding** - Text query converted to embedding via Sentence-Transformers
2. **Vector Search** - ChromaDB queries "video_transcripts" collection
3. **Diversity Re-ranking** - Temporal diversity applied to transcript chunks
4. **Result Display** - Matching text excerpts with timestamps and similarity scores

**Combined Search:**
- Runs both pipelines in parallel
- Returns separate frame and transcript results
- Enables multimodal queries (e.g., "red balloon" finds visual + mentions)

## Key Features

### Multimodal Search
Search across both visual content and spoken words:
- **Frame search** - Find visual content (objects, scenes, people)
- **Transcript search** - Find spoken words and phrases
- **Combined search** - Query both modalities simultaneously

### Scene-Aware Diversity (Frames)
Search results are diversified to show frames from different scenes rather than clustering similar sequential frames. Falls back to temporal diversity if scene detection is disabled. Configurable diversity weight (0=relevance only, 1=diversity only).

### Temporal Diversity (Transcripts)
Transcript results spread across different time periods to avoid clustering similar content from the same conversation segment. Uses 20-second normalization window.

### Solid Frame Detection
Conservative filtering to skip truly blank frames:
- Nearly-black or nearly-white detection
- Low standard deviation (solid colors)
- Edge detection (Canny) for minimal detail frames

### Scene Detection
Identifies visual discontinuities between frames:
- Compares HSV histograms across color channels
- Configurable sensitivity threshold (default 30.0)
- Assigns scene IDs for diversity-aware search

### Speech Transcription
Faster-Whisper audio transcription:
- 4x faster than openai-whisper with lower memory usage
- Automatic language detection
- Segment-level timestamps for precise alignment
- Voice Activity Detection (VAD) to filter silence
- 20-second chunk duration for context balance
- Supports 99+ languages
- GPU acceleration with CTranslate2

## Configuration

**Frame Extraction:**
- Interval: 0.5-5.0 seconds (default 1.5s)
- Solid frame filtering: on/off

**Scene Detection:**
- Threshold: 0-100+ (default 30.0, lower = more sensitive)

**Transcription:**
- Faster-Whisper model: tiny/base/small/medium/large-v3 (default: base)
- Chunk duration: 20 seconds (fixed)
- VAD filtering: enabled by default
- Enable/disable transcription

**Search:**
- Search type: frames/transcripts/both
- Results: 1-20 per type
- Diversity weight: 0-1 (default 0.5)
- Filter by specific video or search all

## Data Storage

- **Vector DB:** ChromaDB persistent store at `./chroma_db/`
  - **Collection: "video_frames"** - Frame embeddings with visual metadata
  - **Collection: "video_transcripts"** - Transcript embeddings with text metadata
- **Frame Cache:** Extracted frames saved to `frame_output/` for result reconstruction

**Metadata per frame:**
- video_id, timestamp, frame_index, video_path, scene_id

**Metadata per transcript chunk:**
- video_id, start_time, end_time, text, video_path

## Current Capabilities

- Multi-video indexing with progress tracking
- Natural language semantic search across visual frames and transcripts
- Speech-to-text transcription with Whisper
- Scene change detection with histogram analysis
- Temporal and scene-based result diversity
- Separate and combined search modes (frames/transcripts/both)
- GPU acceleration with CPU fallback
- Tabbed web interface for upload and search
- Database management (clear, frame/transcript count)
