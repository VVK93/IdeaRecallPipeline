import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LLM Configuration ---
# Load keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Identifiers
GENERATOR_MODEL_ID = "gpt-4-turbo" # Or specific version like "gpt-4-1106-preview"
JUDGE_MODEL_ID = "gemini-1.5-flash-latest"

# --- Evaluation Thresholds ---
BERTSCORE_THRESHOLD = 0.55 # Minimum F1 score for semantic sanity check
SUMMARY_MAX_TOKENS = 500 # Target constraint given to LLM
FLASHCARDS_MAX_TOKENS = 150 # Target constraint given to LLM
ACCURACY_FAILURE_THRESHOLD_PERCENT = 5.0 # Target: Less than 5% inaccurate (for overall tracking)
COMPLETENESS_TARGET_SCORE = 4.0
RELEVANCE_TARGET_SCORE = 4.0
CLARITY_TARGET_SCORE = 4.0
UTILITY_TARGET_SCORE = 3.8 # Target for user ratings

# --- Prompts ---

OUTPUT_JSON_STRUCTURE_DESCRIPTION = """
The output MUST be a valid JSON object containing exactly two keys: "summary" and "flashcards".
- "summary": A string containing the comprehensive summary text.
- "flashcards": A JSON array of objects. Each object MUST have two keys: "question" (string) and "answer" (string).

Example Format:
{
  "summary": "The video explains concept X, detailing A, B, and C. It compares X to Y and concludes Z.",
  "flashcards": [
    {"question": "What is the main definition of X?", "answer": "X is defined as..."},
    {"question": "How does X compare to Y according to the video?", "answer": "X is faster/cheaper/different than Y because..."},
    {"question": "What was the main conclusion regarding Z?", "answer": "The main conclusion was that Z leads to..."}
  ]
}
"""

SYSTEM_PROMPT_GENERATION = f"""You are an expert content analyzer tasked with creating concise, clear, and accurate summaries and effective learning flashcards from video transcripts. Your goal is to help users quickly understand core concepts and practice recall.

**Instructions & Constraints:**

1.  **Accuracy & Faithfulness:** Base the summary and flashcards *strictly* on the information present in the provided transcript. Do *not* add external information, opinions, interpretations, or speculations. Avoid hallucinations.
2.  **Completeness:** Identify and include *all* essential main ideas, key arguments, and significant conclusions presented in the transcript within the summary. Ensure flashcards cover distinct key concepts.
3.  **Relevance:** Focus *only* on the most important, central topics for both the summary and flashcards. Omit minor details, tangential information, and redundancy unless critical for understanding the core message.
4.  **Length:**
    *   The entire "summary" text MUST be **under {SUMMARY_MAX_TOKENS} tokens**.
    *   The *entire* "flashcards" array (all questions and answers combined) MUST total **under {FLASHCARDS_MAX_TOKENS} tokens**. Aim for 5-10 high-quality flashcards.
5.  **Clarity:** Write the summary in clear, fluent, and easily understandable language. Flashcard questions should be unambiguous, and answers should be direct and concise.
6.  **Utility Focus:** Ensure the summary provides actionable understanding and the flashcards are well-suited for active recall practice (clear Q&A targeting key info).

**Output Format:**
ALWAYS respond with a single, valid JSON object matching the exact structure specified below. Do not include any text outside the JSON structure. Ensure the output is **only the JSON object itself**.
{OUTPUT_JSON_STRUCTURE_DESCRIPTION}
Adhere STRICTLY to the JSON format.
"""

USER_PROMPT_GENERATION = """Please analyze the following transcript and generate a JSON output containing:
1. A summary that is accurate, complete, relevant, clear, and adheres to the length limits.
2. A set of 5-10 flashcards that are accurate, relevant, clear, useful for recall, and adhere to the length limits.

Strictly follow the JSON format specified in the system instructions. Respond ONLY with the valid JSON object.

Transcript:
\"\"\"
{}
\"\"\""""

# AI Judge Prompt (Using the combined one)
# Note: We will construct the final prompt content in llm_interface.py for Gemini
AI_JUDGE_PROMPT_TEMPLATE = f"""You are an impartial and meticulous AI Quality Assurance evaluator. Your task is to critically assess AI-generated text (a summary or flashcard set) based on its corresponding source transcript, evaluating it against several specific criteria. Provide your evaluation only in the specified JSON format.

Evaluation Criteria & Instructions:

Carefully evaluate the "Generated Text" based only on the "Source Transcript" according to the criteria below. Evaluate each criterion independently before constructing the final JSON output.

1.  Accuracy & Faithfulness:
    *   Task: Compare the "Generated Text" against the "Source Transcript". Identify if it contains any significant factual inaccuracies, statements contradicting the source, or information clearly not present in the source (hallucinations). Minor phrasing differences preserving meaning are acceptable.
    *   Output Field (`contains_inaccuracies`): A boolean value (`true` if inaccuracies/hallucinations are found, `false` otherwise).
    *   Output Field (`accuracy_explanation`): If `contains_inaccuracies` is `true`, provide a brief explanation citing the inaccurate statement(s). If `false`, leave this as an empty string `""`.

2.  Completeness:
    *   Task: Assess if the "Generated Text" captures all the essential main ideas, key arguments, and significant conclusions presented in the "Source Transcript".
    *   Scale: 1 (Very Incomplete) to 5 (Fully Comprehensive).
    *   Output Field (`completeness_score`): An integer score from 1 to 5.

3.  Relevance:
    *   Task: Evaluate how well the "Generated Text" focuses on the *most important, core topics* of the "Source Transcript", filtering out minor details, tangents, or excessive examples.
    *   Scale: 1 (Mostly Irrelevant) to 5 (Highly Relevant to Core Topics).
    *   Output Field (`relevance_score`): An integer score from 1 to 5.

4.  Clarity:
    *   Task: Assess the clarity, fluency, and ease of understanding of the "Generated Text" itself. Consider grammar, sentence structure, and word choice.
    *   Scale: 1 (Very Unclear) to 5 (Very Clear and Readable).
    *   Output Field (`clarity_score`): An integer score from 1 to 5.

Source Transcript
{{transcript}}

Input 2: Generated Text to Evaluate
{{generated_json_string}}

Required Output Format:
Respond only with a single JSON object containing your evaluation results. Do not include any introductory text, explanations outside the JSON structure, or markdown formatting around the JSON block."""
