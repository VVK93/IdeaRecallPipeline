import config
import json
import time
import openai
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted

# --- Configure LLM Clients ---
openai_client = None
if config.OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")

gemini_model = None
if config.GOOGLE_API_KEY:
    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        # Set safety settings to minimum to avoid blocking potentially relevant content
        # Adjust these based on your content policy needs
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        gemini_model = genai.GenerativeModel(
            config.JUDGE_MODEL_ID,
            safety_settings=safety_settings
            )
    except Exception as e:
        print(f"Error initializing Google Gemini client: {e}")

# --- LLM Interaction Functions ---

def call_generator_llm(transcript):
    """Calls the Generator LLM (GPT-4 Turbo) to get summary and flashcards."""
    if not openai_client:
        return None, "OpenAI client not initialized. Check API key."

    print(f"--- Calling Generator LLM: {config.GENERATOR_MODEL_ID} ---")
    start_time = time.time()
    try:
        response = openai_client.chat.completions.create(
            model=config.GENERATOR_MODEL_ID,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT_GENERATION},
                {"role": "user", "content": config.USER_PROMPT_GENERATION.format(transcript)}
            ],
            temperature=0.5, # Lower temperature for more focused output
            response_format={"type": "json_object"} # Request JSON output
        )
        duration = time.time() - start_time
        print(f"--- Generator LLM call took {duration:.2f}s ---")

        result_json_str = response.choices[0].message.content
        # Further validation to ensure it's *just* the JSON might be needed
        # if the model occasionally adds surrounding text despite instructions
        try:
            # Attempt to parse to validate early
            json.loads(result_json_str)
            return result_json_str, None # Return raw JSON string and no error
        except json.JSONDecodeError as json_err:
             print(f"(!) Generator LLM response was not valid JSON: {json_err}")
             print(f"Raw Response: {result_json_str[:500]}...") # Log snippet
             return result_json_str, f"Generator LLM response was not valid JSON: {json_err}"


    except openai.APIConnectionError as e:
        print(f"OpenAI API connection error: {e}")
        return None, f"OpenAI API connection error: {e}"
    except openai.RateLimitError as e:
        print(f"OpenAI API rate limit exceeded: {e}")
        return None, f"OpenAI API rate limit exceeded: {e}"
    except openai.AuthenticationError as e:
         print(f"OpenAI Authentication Error: Check API Key. {e}")
         return None, f"OpenAI Authentication Error: Check API Key. {e}"
    except openai.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        return None, f"OpenAI API returned an API Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred calling Generator LLM: {e}")
        return None, f"An unexpected error occurred: {e}"


def call_ai_judge_llm(transcript, generated_json_string):
    """Calls the AI Judge LLM (Gemini 1.5 Flash) to evaluate the generated text."""
    if not gemini_model:
        return None, "Gemini client not initialized. Check API key."

    print(f"--- Calling AI Judge LLM: {config.JUDGE_MODEL_ID} ---")
    start_time = time.time()

    # Construct the prompt for Gemini
    prompt_content = config.AI_JUDGE_PROMPT_TEMPLATE.format(
        transcript=transcript,
        generated_json_string=generated_json_string
    )

    # Specify JSON output format for Gemini 1.5 models
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.2 # Low temperature for consistent evaluation
        )

    try:
        response = gemini_model.generate_content(
            prompt_content,
            generation_config=generation_config,
            request_options={'timeout': 120} # Set a timeout in seconds
        )

        duration = time.time() - start_time
        print(f"--- AI Judge LLM call took {duration:.2f}s ---")

        # Check for safety blocks before accessing text
        if not response.candidates:
             block_reason = response.prompt_feedback.block_reason
             print(f"(!) AI Judge response blocked. Reason: {block_reason}")
             return None, f"AI Judge response blocked. Reason: {block_reason}"
             
        result_json_str = response.text
        # Validate JSON early
        try:
             json.loads(result_json_str)
             return result_json_str, None # Return raw JSON string, no error
        except json.JSONDecodeError as json_err:
             print(f"(!) AI Judge response was not valid JSON: {json_err}")
             print(f"Raw Response: {result_json_str[:500]}...") # Log snippet
             return result_json_str, f"AI Judge response was not valid JSON: {json_err}"

    except ResourceExhausted as e:
         print(f"Google API Resource Exhausted (Rate Limit?): {e}")
         return None, f"Google API Resource Exhausted (Rate Limit?): {e}"
    except GoogleAPIError as e:
         print(f"Google API Error occurred: {e}")
         return None, f"Google API Error occurred: {e}"
    except Exception as e:
        print(f"An unexpected error occurred calling AI Judge LLM: {e}")
        return None, f"An unexpected error occurred: {e}"