# AI Product Evaluation Pipeline

This is a code for Streamlit-based application that implements the evaluation pipeline for Ideas Recall telegram bot and fully described in this article https://vvk93.substack.com/p/building-an-ai-powered-telegram-bot
This evaluation pipeline was used to test prompts and AI capabilities and was later used for real telegram bot with some minor changes.

## Overview

This project implements an automated evaluation pipeline that:
1. Downloads transcripts from YouTube videos
2. Generates summaries and flashcards using AI
3. Evaluates the quality of generated content through multiple stages
4. Provides detailed metrics and human feedback capabilities

## Main Features

### 1. YouTube Transcript Processing
- Automatic transcript download from YouTube videos
- Support for various video formats and languages

### 2. AI-Powered Content Generation
- Generates concise summaries from video transcripts
- Creates educational flashcards for key concepts
- Uses configurable AI models for generation

### 3. Multi-Stage Evaluation Pipeline
- **Stage 1: Automated Checks**
  - JSON format validation
  - Length verification
  - BERTScore semantic similarity check

- **Stage 2: AI Judge Assessment**
  - Accuracy evaluation
  - Completeness scoring
  - Relevance assessment
  - Clarity measurement

- **Stage 3: Human Feedback**
  - User utility rating system
  - Interactive feedback collection

### 4. Comprehensive Dashboard
- Real-time pipeline execution status
- Detailed evaluation metrics
- Historical run logs
- Configuration management
- Raw output inspection

## Technical Stack

- **Frontend**: Streamlit
- **AI Models**: OpenAI API
- **Video Processing**: YouTube API
- **Evaluation Metrics**: BERTScore, Custom AI Judge

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GOOGLE_API_KEY`: Your Google API key

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Configuration

The application allows customization of:
- AI model selection
- Evaluation thresholds
- Token limits
- System prompts
- Target scores

## Usage

1. Enter a YouTube URL in the sidebar
2. Click "Download & Run Pipeline"
3. View results across multiple tabs:
   - Configuration & Prompts
   - Generated Output
   - Evaluation Results
   - Run Log & History
