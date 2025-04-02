import json
from bert_score import score as bert_score_calculate
import torch
import tiktoken 
import config
import transformers # Needed dependency for bert-score
import logging

logger = logging.getLogger(__name__) 

# --- Token Counting ---
try:
    encoding = tiktoken.get_encoding("cl100k_base")
    logger.info("Tiktoken encoding 'cl100k_base' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load tiktoken encoding 'cl100k_base': {e}", exc_info=True)
    encoding = None

def count_tokens(text: str) -> int:
    """Counts the number of tokens in a text string using tiktoken."""
    if encoding is None:
        logger.error("Tiktoken encoding not available, cannot count tokens accurately.")
        return len(text.split()) # Fallback
    if not isinstance(text, str):
        logger.warning(f"Input to count_tokens was not a string (type: {type(text)}), returning 0 tokens.")
        return 0
    try:
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        logger.error(f"Error during tiktoken encoding: {e}", exc_info=True)
        return -1

# --- Format and Structure Checks (Keep previous version) ---
def check_format(generated_json_str):
    """Checks if the string is valid JSON and has the required top-level keys."""
    logger.debug("Checking format of generated JSON string...")
    if not generated_json_str:
        logger.warning("Format check failed: Generated output is empty.")
        return False, "Generated output is empty"
    try:
        data = json.loads(generated_json_str)
        if isinstance(data, dict) and "summary" in data and "flashcards" in data:
            if isinstance(data["summary"], str) and isinstance(data["flashcards"], list):
                 for i, item in enumerate(data["flashcards"]):
                     if not (isinstance(item, dict) and "question" in item and "answer" in item and
                             isinstance(item["question"], str) and isinstance(item["answer"], str) ):
                         msg = f"Invalid flashcard structure or types in item {i}: {item}"
                         logger.warning(f"Format check failed: {msg}")
                         return False, msg
                 logger.info("Format check passed.")
                 return True, "Valid format"
            else:
                msg = f"Incorrect types for summary (type: {type(data.get('summary'))}) or flashcards (type: {type(data.get('flashcards'))})"
                logger.warning(f"Format check failed: {msg}")
                return False, msg
        else:
            msg = f"Missing required keys ('summary', 'flashcards'). Found keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
            logger.warning(f"Format check failed: {msg}")
            return False, msg
    except json.JSONDecodeError as e:
        logger.warning(f"Format check failed: Invalid JSON - {e}")
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during format check: {e}", exc_info=True)
        return False, f"Unexpected error during format check: {e}"


def check_length(generated_data):
    """Checks token limits for summary and flashcards using tiktoken."""
    logger.debug("Performing length check using tiktoken...")
    if encoding is None:
        logger.error("Cannot perform accurate length check: tiktoken encoding not loaded.")
        return False, {"error": "Tokenizer not available"}

    summary_text = generated_data.get("summary", "")
    summary_tokens = count_tokens(summary_text)

    flashcards_text = ""
    flashcards_list = generated_data.get("flashcards", [])
    if isinstance(flashcards_list, list):
        for card in flashcards_list:
            q = card.get("question", "")
            a = card.get("answer", "")
            flashcards_text += f"Q: {q}\nA: {a}\n\n"
    else:
        logger.warning(f"Flashcards data is not a list (type: {type(flashcards_list)}), cannot calculate token count.")

    flashcards_tokens = count_tokens(flashcards_text.strip())

    summary_ok = summary_tokens >= 0 and summary_tokens <= config.SUMMARY_MAX_TOKENS
    flashcards_ok = flashcards_tokens >= 0 and flashcards_tokens <= config.FLASHCARDS_MAX_TOKENS

    details = {
        "summary_tokens": summary_tokens if summary_tokens >= 0 else "Error",
        "flashcards_tokens": flashcards_tokens if flashcards_tokens >= 0 else "Error",
        "summary_limit": config.SUMMARY_MAX_TOKENS,
        "flashcards_limit": config.FLASHCARDS_MAX_TOKENS,
        "summary_ok": summary_ok,
        "flashcards_ok": flashcards_ok,
        "overall_ok": summary_ok and flashcards_ok
    }
    logger.info(f"Length check results: {details}")
    return details["overall_ok"], details

# --- Semantic Similarity ---

def calculate_bertscore(generated_summary, reference_transcript):
    """Calculates BERTScore F1 between generated summary and reference transcript."""
    logger.debug("Attempting to calculate BERTScore...")
    if not generated_summary or not reference_transcript:
        logger.warning("Skipping BERTScore calculation: Missing summary or transcript.")
        return 0.0, "Missing summary or transcript for BERTScore"

    try:
        # Log input snippets for debugging
        summary_snippet = str(generated_summary)[:100] + "..." if len(str(generated_summary)) > 100 else str(generated_summary)
        ref_snippet = str(reference_transcript)[:100] + "..." if len(str(reference_transcript)) > 100 else str(reference_transcript)
        logger.debug(f"BERTScore Inputs: Cand='{summary_snippet}', Ref='{ref_snippet}'")

        cands = [str(generated_summary)]
        refs = [str(reference_transcript)]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Calculating BERTScore on device: {device}")

        if 'transformers' not in globals():
            raise ImportError("Transformers library is required for BERTScore but not found.")

        # Calculate RAW scores first (remove rescale_with_baseline)
        P, R, F1 = bert_score_calculate(cands, refs, lang="en", verbose=False, device=device) # Removed rescale_with_baseline=True

        # Log the raw tensors
        logger.debug(f"Raw BERTScore Tensors: P={P}, R={R}, F1={F1}")

        # Extract F1 score
        f1_score = F1.item()

        # Add a check for unexpected range
        if not (0.0 <= f1_score <= 1.0):
             logger.warning(f"BERTScore F1 calculation resulted in an unexpected value: {f1_score:.4f}. Check inputs and library version.")
             # You might choose to clamp it or return an error indicator
             # Clamping example: f1_score = max(0.0, min(1.0, f1_score))
             # For now, just log the warning.

        logger.info(f"BERTScore F1 calculated (raw): {f1_score:.4f}")
        return f1_score, f"BERTScore F1: {f1_score:.4f}"

    except ImportError as e:
         logger.error(f"ImportError during BERTScore calculation: {e}. Is 'transformers' installed?")
         return 0.0, f"ImportError: {e}"
    except Exception as e:
        # Log the specific inputs that caused the error if possible
        logger.error(f"Error calculating BERTScore for Cand='{summary_snippet}', Ref='{ref_snippet}': {e}", exc_info=True)
        return 0.0, f"Error calculating BERTScore: {e}"


# --- AI Judge Parsing (Keep previous version that handles flattened structure) ---
def parse_ai_judge_response(response_str):
    """
    Parses the JSON response from the AI Judge.
    Handles both the intended nested structure and a common flattened deviation.
    """
    logger.debug("Attempting to parse AI Judge response...")
    if not response_str:
        logger.error("AI Judge response string is empty.")
        return None, "AI Judge response was empty"
    try:
        original_response_str = response_str
        if response_str.strip().startswith("```json"):
             response_str = response_str.strip()[7:-3].strip()
             logger.debug("Stripped ```json markdown wrapper.")
        elif response_str.strip().startswith("```"):
             response_str = response_str.strip()[3:-3].strip()
             logger.debug("Stripped ``` markdown wrapper.")

        eval_data = json.loads(response_str)
        logger.debug(f"Successfully loaded JSON from AI Judge: {eval_data}")

        required_top_level_keys = ["completeness_score", "relevance_score", "clarity_score"]
        reconstructed_data = {}
        missing_keys = []

        for key in required_top_level_keys:
            if key not in eval_data:
                missing_keys.append(key)
            else:
                reconstructed_data[key] = eval_data[key]

        if "accuracy_assessment" in eval_data and isinstance(eval_data["accuracy_assessment"], dict) and "contains_inaccuracies" in eval_data["accuracy_assessment"]:
            logger.debug("Found expected nested 'accuracy_assessment' structure.")
            reconstructed_data["accuracy_assessment"] = {
                "contains_inaccuracies": eval_data["accuracy_assessment"].get("contains_inaccuracies"),
                "explanation": eval_data["accuracy_assessment"].get("explanation", "")
            }
        elif "contains_inaccuracies" in eval_data:
             logger.warning("Found flattened accuracy structure from AI Judge. Reconstructing to expected nested format.")
             reconstructed_data["accuracy_assessment"] = {
                 "contains_inaccuracies": eval_data.get("contains_inaccuracies"),
                 "explanation": eval_data.get("accuracy_explanation", eval_data.get("explanation", ""))
             }
        else:
            missing_keys.append("accuracy_assessment (or contains_inaccuracies)")

        if missing_keys:
            error_msg = f"AI Judge JSON missing required keys: {', '.join(missing_keys)}"
            logger.error(error_msg + f" - Parsed Data: {eval_data}")
            return None, error_msg

        reconstructed_data["optional_overall_notes"] = eval_data.get("optional_overall_notes", "")

        logger.info("Successfully parsed and validated AI Judge response (handling structure variations).")
        return reconstructed_data, "Successfully parsed AI Judge response"

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON from AI Judge: {e}"
        logger.error(error_msg + f" - Raw response snippet: {original_response_str[:500]}...")
        return None, error_msg
    except Exception as e:
         error_msg = f"Unexpected error parsing AI Judge response: {e}"
         logger.error(error_msg + f" - Raw response snippet: {original_response_str[:500]}...", exc_info=True)
         return None, error_msg