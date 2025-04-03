import streamlit as st
import json
import time
import pandas as pd
import logging
import traceback
import youtube_handler

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler("pipeline.log", mode='a')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        st.error(f"Could not configure file logging: {e}")
        logger.error(f"Failed to configure file logger: {e}")

logger.info("--- Starting Streamlit App ---")

try:
    import config
    import llm_interface
    import evaluation_metrics
    logger.info("Successfully imported config, llm_interface, and evaluation_metrics.")
except ImportError as e:
     logger.critical(f"Failed to import necessary modules: {e}", exc_info=True)
     st.error(f"Critical Error: Failed to import necessary modules ({e}). Check file structure and dependencies.")
     st.stop()
except Exception as e:
     logger.critical(f"An unexpected error occurred during imports: {e}", exc_info=True)
     st.error(f"Critical Error during imports: {e}")
     st.stop()

st.set_page_config(layout="wide", page_title="AI Product Eval Pipeline")

def display_evaluation_results(eval_data):
    if not eval_data:
        st.info("No evaluation data available for the selected run.")
        return

    log_timestamp = eval_data.get('timestamp')
    logger.debug(f"Displaying evaluation results for timestamp: {log_timestamp}")

    st.markdown("---")

    gen_time = eval_data.get("generation_time", 0)
    st.subheader(f"Generation Stage 🎯 ({gen_time:.1f}s)")
    if eval_data.get("generator_error"):
        st.error(f"Generation Failed: {eval_data.get('generator_error')}")
    else:
        st.success("Generation completed successfully")

    st.markdown("---")

    stage1_time = eval_data.get("stage1_time", 0)
    st.subheader(f"Stage 1: Automated Checks ⚡ ({stage1_time:.1f}s)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        format_check_data = eval_data.get("format_check", {})
        format_ok = format_check_data.get("passed", False)
        format_msg = format_check_data.get("message", 'N/A')
        st.metric(label="JSON Format Check", value=f"{'✅ PASS' if format_ok else '❌ FAIL'}", delta_color="off",
                  help=format_msg if not format_ok else None)
        if not format_ok:
            st.error(f"Reason: {format_msg}")
            
    with col2:
        length_check_data = eval_data.get("length_check", {})
        length_ok = length_check_data.get("passed", False)
        len_details = length_check_data.get("details", {})
        st.metric(label="Length Check", value=f"{'✅ PASS' if length_ok else '❌ FAIL'}", delta_color="off")
        if not length_ok:
            if 'error' in len_details:
                 st.error(f"Reason: {len_details['error']}")
            else:
                if not len_details.get('summary_ok', True):
                    st.warning(f"Summary: {len_details.get('summary_tokens', 'N/A')} / {len_details.get('summary_limit', 'N/A')} tokens")
                if not len_details.get('flashcards_ok', True):
                    st.warning(f"Flashcards: {len_details.get('flashcards_tokens', 'N/A')} / {len_details.get('flashcards_limit', 'N/A')} tokens")
        elif len_details:
             st.caption(f"Summary: {len_details.get('summary_tokens', 'N/A')} / {len_details.get('summary_limit', 'N/A')}")
             st.caption(f"Flashcards: {len_details.get('flashcards_tokens', 'N/A')} / {len_details.get('flashcards_limit', 'N/A')}")

    with col3:
        bert_score_data = eval_data.get("bert_score", {})
        bert_score = bert_score_data.get("score", None)
        bert_passed = bert_score_data.get("passed_threshold", False)
        bert_msg = bert_score_data.get("message","")
        st.metric(label="BERTScore F1 (Sanity Check)", value=f"{'✅' if bert_passed else '❌'} {bert_score:.3f}" if bert_score is not None else "N/A",
                  help=f"Compares summary to transcript. Low score might indicate semantic detachment.\n{bert_msg}")
        if bert_score is not None:
            st.caption(f"Threshold: {config.BERTSCORE_THRESHOLD} - {'Passed' if bert_passed else 'Below Threshold'}")
            if not bert_passed and bert_score > 0.0:
                st.warning("Low semantic relevance detected.")
        else:
             st.caption("Not calculated (e.g., missing summary)")

    st.markdown("---")

    stage2_time = eval_data.get("stage2_time", 0)
    st.subheader(f"Stage 2: AI Judge Assessment 🤖 ({stage2_time:.1f}s)")
    ai_judge_results_data = eval_data.get("ai_judge_assessment", {})
    ai_judge_results = ai_judge_results_data.get("data")
    ai_judge_error = ai_judge_results_data.get("error")
    ai_judge_raw = ai_judge_results_data.get("raw_response")

    if ai_judge_error:
        st.error(f"AI Judge Error/Skip Reason: {ai_judge_error}")
        if ai_judge_raw:
             with st.expander("Show Raw AI Judge Response"):
                  st.code(ai_judge_raw, language="text")

    elif ai_judge_results:
        col_acc, col_comp, col_rel, col_clar = st.columns(4)
        
        accuracy_data = ai_judge_results.get("accuracy_assessment", {})
        inaccurate = accuracy_data.get("contains_inaccuracies", None)
        with col_acc:
            st.markdown("**Accuracy**")
            if inaccurate is None:
                 st.json({"Result": "N/A"})
            else:
                result = "✅ Accurate" if not inaccurate else "❌ Inaccurate"
                st.json({"Result": result})
            if inaccurate:
                st.error(f"Explanation: {accuracy_data.get('explanation', 'None provided')}", icon="❗")
                
        comp_score = ai_judge_results.get("completeness_score")
        comp_target = config.COMPLETENESS_TARGET_SCORE
        with col_comp:
             st.markdown("**Completeness**")
             st.metric("Score (AI)", f"{'✅' if comp_score >= comp_target else '❌'} {comp_score}/5" if comp_score else "N/A",
                      delta=f"{comp_score - comp_target:.1f}" if isinstance(comp_score, (int, float)) else None,
                      delta_color="normal" if isinstance(comp_score, (int, float)) and comp_score >= comp_target else "inverse",
                      help=f"Target: ≥ {comp_target}")
                      
        rel_score = ai_judge_results.get("relevance_score")
        rel_target = config.RELEVANCE_TARGET_SCORE
        with col_rel:
             st.markdown("**Relevance**")
             st.metric("Score (AI)", f"{'✅' if rel_score >= rel_target else '❌'} {rel_score}/5" if rel_score else "N/A",
                      delta=f"{rel_score - rel_target:.1f}" if isinstance(rel_score, (int, float)) else None,
                      delta_color="normal" if isinstance(rel_score, (int, float)) and rel_score >= rel_target else "inverse",
                      help=f"Target: ≥ {rel_target}")
                      
        clar_score = ai_judge_results.get("clarity_score")
        clar_target = config.CLARITY_TARGET_SCORE
        with col_clar:
            st.markdown("**Clarity**")
            st.metric("Score (AI)", f"{'✅' if clar_score >= clar_target else '❌'} {clar_score}/5" if clar_score else "N/A",
                     delta=f"{clar_score - clar_target:.1f}" if isinstance(clar_score, (int, float)) else None,
                     delta_color="normal" if isinstance(clar_score, (int, float)) and clar_score >= clar_target else "inverse",
                     help=f"Target: ≥ {clar_target}")

        notes = ai_judge_results.get("optional_overall_notes")
        if notes:
            st.info(f"AI Judge Notes: {notes}", icon="ℹ️")
    else:
        st.warning("AI Judge assessment was not performed or did not return valid data.")

    st.markdown("---")

    st.subheader(f"Stage 3: Human Feedback (Utility) 👤")
    user_rating = eval_data.get("user_utility_rating")
    util_target = config.UTILITY_TARGET_SCORE
    if user_rating:
        st.metric("User Utility Rating", f"{'✅' if user_rating >= util_target else '❌'} {user_rating}/5",
                   delta=f"{user_rating - util_target:.1f}" if isinstance(user_rating, (int, float)) else None,
                   delta_color="normal" if isinstance(user_rating, (int, float)) and user_rating >= util_target else "inverse",
                   help=f"Target: ≥ {util_target}")
    else:
        st.info("User rating not yet provided for this item.")

keys_ok = True
try:
    if not config.OPENAI_API_KEY:
        st.error("OpenAI API Key not found. Please set it in the .env file (OPENAI_API_KEY=...)")
        logger.error("OpenAI API Key not found in environment variables.")
        keys_ok = False
    if not config.GOOGLE_API_KEY:
        st.error("Google API Key not found. Please set it in the .env file (GOOGLE_API_KEY=...)")
        logger.error("Google API Key not found in environment variables.")
        keys_ok = False
except AttributeError:
     st.error("Could not access API keys from config. Check config.py and .env file.")
     logger.critical("AttributeError accessing API keys in config. Stopping.")
     keys_ok = False
     st.stop()
except Exception as e:
    st.error(f"Unexpected error checking API keys: {e}")
    logger.critical(f"Unexpected error checking API keys: {e}", exc_info=True)
    keys_ok = False
    st.stop()

st.title("📝 Evaluation Pipeline for Idea Recall Bot")

st.sidebar.header("Input")

youtube_url = st.sidebar.text_input(
    "YouTube Video URL:",
    key="youtube_url",
    help="Enter a YouTube video URL to automatically download its transcript and run the evaluation pipeline."
)

if st.sidebar.button("🚀 Download & Run Pipeline", disabled=not youtube_url):
    with st.spinner("Downloading transcript and running pipeline..."):
        transcript, error = youtube_handler.download_transcript(youtube_url)
        if error:
            st.sidebar.error(f"Failed to download transcript: {error}")
        else:
            st.sidebar.success("Transcript downloaded successfully!")
            st.session_state.transcript_input = transcript
            st.session_state.run_pipeline = True

if 'transcript_input' not in st.session_state:
    st.session_state.transcript_input = ""
if 'run_pipeline' not in st.session_state:
    st.session_state.run_pipeline = False

st.sidebar.divider()
st.sidebar.header("Iterative Improvement Notes")
st.sidebar.info(
    "Analyze results in the 'Evaluation Results' and 'Run Log' tabs. If metrics fall below targets, consider refining prompts (see 'Configuration' tab), adjusting thresholds, or changing models."
)
st.sidebar.caption(f"Generator: {config.GENERATOR_MODEL_ID}\nJudge: {config.JUDGE_MODEL_ID}")

if 'evaluation_log' not in st.session_state:
    st.session_state.evaluation_log = []
    logger.debug("Initialized evaluation_log in session state.")
if 'current_run_index' not in st.session_state:
    st.session_state.current_run_index = None
    logger.debug("Initialized current_run_index in session state.")

if st.session_state.run_pipeline:
    run_timestamp = time.time()
    logger.info(f"--- Pipeline Run Initiated: Timestamp {run_timestamp} ---")
    processing_placeholder = st.empty()
    processing_placeholder.info("🚀 Pipeline Run Initiated...")

    current_eval_data = {"transcript": st.session_state.transcript_input, "timestamp": run_timestamp}
    evaluation_results = {"timestamp": run_timestamp}

    st.session_state.run_pipeline = False

    try:
        logger.info("Initiating Generation Stage.")
        gen_start_time = time.time()
        with processing_placeholder:
             with st.spinner(f"Running Pipeline... Calling Generator ({config.GENERATOR_MODEL_ID})..."):
                generated_json_str, gen_error = llm_interface.call_generator_llm(st.session_state.transcript_input)
                current_eval_data["generated_json_str"] = generated_json_str
                current_eval_data["generator_error"] = gen_error
                logger.debug(f"Generator Raw Response Snippet: {generated_json_str[:200] if generated_json_str else 'None'}...")
                if gen_error: logger.error(f"Generator Error Occurred: {gen_error}")
        gen_time = time.time() - gen_start_time
        evaluation_results["generation_time"] = gen_time

        if gen_error or not generated_json_str:
            error_msg = gen_error or "Empty response from Generator LLM."
            st.error(f"Generation Failed: {error_msg}")
            if generated_json_str: st.code(generated_json_str, language='text')
            current_eval_data["evaluation_results"] = evaluation_results
            st.session_state.evaluation_log.append(current_eval_data)
            st.session_state.current_run_index = len(st.session_state.evaluation_log) - 1
            processing_placeholder.empty()
            st.stop()

        logger.info("Checking format of generated output.")
        format_passed, format_msg = evaluation_metrics.check_format(generated_json_str)
        evaluation_results["format_check"] = {"passed": format_passed, "message": format_msg}
        generated_data = None
        if format_passed:
            logger.info("Generated output format check passed.")
            try:
                generated_data = json.loads(generated_json_str)
                current_eval_data["generated_data"] = generated_data
                logger.debug("Successfully parsed generated data.")
            except Exception as e:
                format_passed = False
                format_msg = f"Passed initial check but failed json.loads: {e}"
                evaluation_results["format_check"] = {"passed": format_passed, "message": format_msg}
                current_eval_data["generated_data"] = {"error": f"JSON Parsing Error: {e}"}
                logger.error(f"Error parsing JSON even after format check passed: {e}", exc_info=True)
        else:
            current_eval_data["generated_data"] = {"error": "Invalid JSON"}
            logger.error(f"Generated output failed format check: {format_msg}")

        logger.info("Starting Evaluation Stages.")

        logger.info("Running Stage 1: Length & BERTScore Checks.")
        stage1_start_time = time.time()
        with processing_placeholder:
            with st.spinner("Running Stage 1: Length & BERTScore Checks..."):
                if generated_data and "error" not in generated_data:
                    length_passed, length_details = evaluation_metrics.check_length(generated_data)
                    evaluation_results["length_check"] = {"passed": length_passed, "details": length_details}
                    logger.info(f"Length check result: Passed={length_passed}, Details={length_details}")
                else:
                    evaluation_results["length_check"] = {"passed": False, "details": {"error": "Could not parse data for length check"}}
                    logger.warning("Skipping length check due to parsing error or no data.")

                summary_text = generated_data.get("summary") if generated_data and "error" not in generated_data else None
                if summary_text:
                    bert_score_val, bert_msg = evaluation_metrics.calculate_bertscore(summary_text, st.session_state.transcript_input)
                    bert_passed_threshold = bert_score_val >= config.BERTSCORE_THRESHOLD
                    evaluation_results["bert_score"] = {"score": bert_score_val, "message": bert_msg, "passed_threshold": bert_passed_threshold}
                    logger.info(f"BERTScore calculated: Score={bert_score_val:.3f}, PassedThreshold={bert_passed_threshold}")
                else:
                    evaluation_results["bert_score"] = {"score": None, "message": "Summary not available", "passed_threshold": False}
                    logger.warning("Skipping BERTScore calculation as summary is missing or invalid.")
        stage1_time = time.time() - stage1_start_time
        evaluation_results["stage1_time"] = stage1_time

        logger.info("Initiating Stage 2: AI Judge Assessment.")
        stage2_start_time = time.time()
        ai_judge_assessment = {"data": None, "error": None, "raw_response": None}
        run_ai_judge = format_passed
        logger.debug(f"Decision to run AI Judge: {run_ai_judge} (FormatOK={format_passed})")

        if run_ai_judge:
            with processing_placeholder:
                with st.spinner(f"Running Stage 2: AI Judge Assessment ({config.JUDGE_MODEL_ID})..."):
                    logger.info(f"Calling AI Judge: {config.JUDGE_MODEL_ID}")
                    ai_judge_response_str, judge_error = llm_interface.call_ai_judge_llm(st.session_state.transcript_input, generated_json_str)
                    ai_judge_assessment["raw_response"] = ai_judge_response_str
                    logger.debug(f"AI Judge Raw Response Snippet: {ai_judge_response_str[:200] if ai_judge_response_str else 'None'}...")

                    if judge_error:
                         ai_judge_assessment["error"] = judge_error
                         logger.error(f"AI Judge call failed: {judge_error}")
                    elif ai_judge_response_str:
                        logger.info("AI Judge call successful, parsing response.")
                        parsed_judge_data, parse_msg = evaluation_metrics.parse_ai_judge_response(ai_judge_response_str)
                        if parsed_judge_data:
                            ai_judge_assessment["data"] = parsed_judge_data
                            logger.info("Successfully parsed AI Judge response.")
                            logger.debug(f"Parsed AI Judge Data: {parsed_judge_data}")
                        else:
                            ai_judge_assessment["error"] = parse_msg
                            logger.error(f"Failed to parse AI Judge response: {parse_msg}")
                    else:
                         ai_judge_assessment["error"] = "No response from AI Judge LLM."
                         logger.error("AI Judge call returned an empty response.")
        else:
             reason = "Skipped due to failure in prior checks (e.g., invalid format)."
             ai_judge_assessment["error"] = reason
             logger.warning(f"AI Judge assessment skipped. Reason: {reason}")
        stage2_time = time.time() - stage2_start_time
        evaluation_results["stage2_time"] = stage2_time

        evaluation_results["ai_judge_assessment"] = ai_judge_assessment

        logger.info("Setting up Stage 3: Human Feedback placeholder.")
        stage3_start_time = time.time()
        evaluation_results["user_utility_rating"] = None
        stage3_time = time.time() - stage3_start_time
        evaluation_results["stage3_time"] = stage3_time

        current_eval_data["evaluation_results"] = evaluation_results
        st.session_state.evaluation_log.append(current_eval_data)
        st.session_state.current_run_index = len(st.session_state.evaluation_log) - 1
        logger.info(f"Evaluation results stored for timestamp {run_timestamp}. New log length: {len(st.session_state.evaluation_log)}")
        logger.debug(f"Full evaluation results for current run: {evaluation_results}")

    finally:
        processing_placeholder.empty()

    st.rerun()

tab_config, tab_output, tab_eval, tab_log = st.tabs([
    "⚙️ Configuration & Prompts",
    "🤖 Generated Output",
    "📊 Evaluation Results",
    "📜 Run Log & History"
])

with tab_config:
    st.header("Pipeline Configuration")
    st.markdown("**Models Used:**")
    st.text(f"Generator: {config.GENERATOR_MODEL_ID}")
    st.text(f"AI Judge: {config.JUDGE_MODEL_ID}")

    st.markdown("**Evaluation Thresholds & Targets:**")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("BERTScore Threshold", f"≥ {config.BERTSCORE_THRESHOLD:.2f}")
        st.metric("Accuracy Target", f"< {config.ACCURACY_FAILURE_THRESHOLD_PERCENT}% Errors")
        st.metric("Utility Target", f"≥ {config.UTILITY_TARGET_SCORE}/5")
    with col_t2:
        st.metric("Completeness Target", f"≥ {config.COMPLETENESS_TARGET_SCORE}/5")
        st.metric("Relevance Target", f"≥ {config.RELEVANCE_TARGET_SCORE}/5")
        st.metric("Clarity Target", f"≥ {config.CLARITY_TARGET_SCORE}/5")
    with col_t3:
         st.metric("Max Summary Tokens", config.SUMMARY_MAX_TOKENS)
         st.metric("Max Flashcard Tokens", config.FLASHCARDS_MAX_TOKENS)

    st.divider()
    st.header("Prompts Used")
    with st.expander("Generator System Prompt"):
        st.text(config.SYSTEM_PROMPT_GENERATION)
    with st.expander("Generator User Prompt Template"):
        st.text(config.USER_PROMPT_GENERATION)
    with st.expander("AI Judge Prompt Template"):
         judge_prompt_display = config.AI_JUDGE_PROMPT_TEMPLATE.replace('{transcript}', '{transcript_placeholder}').replace('{generated_json_string}', '{generated_json_placeholder}')
         st.text(judge_prompt_display)

    st.divider()
    st.header("Source Transcript")
    if st.session_state.transcript_input:
        with st.expander("📝 View Downloaded Transcript", expanded=False):
            st.text_area("Transcript Text", st.session_state.transcript_input, height=300, disabled=True)
    else:
        st.info("No transcript available. Download a transcript using the YouTube URL input in the sidebar.")

with tab_output:
    st.header("Generated Output")
    if st.session_state.current_run_index is not None:
        current_data = st.session_state.evaluation_log[st.session_state.current_run_index]
        gen_data = current_data.get("generated_data")
        gen_error = current_data.get("generator_error")
        format_error = not current_data.get("evaluation_results", {}).get("format_check", {}).get("passed", True) if current_data.get("evaluation_results") else False

        if gen_error:
             st.error(f"Generation failed: {gen_error}")
             if current_data.get("generated_json_str"):
                 st.code(current_data["generated_json_str"], language="text")
        elif format_error:
             st.error(f"Generated output failed format validation: {current_data.get('evaluation_results', {}).get('format_check', {}).get('message')}")
             st.code(current_data.get("generated_json_str", "No raw string available."), language="text")
        elif gen_data and isinstance(gen_data, dict) and "error" not in gen_data:
            st.markdown(f"Showing output for Run ID: {st.session_state.current_run_index + 1}")
            col1_disp, col2_disp = st.columns(2)
            with col1_disp:
                 st.subheader("Summary")
                 st.markdown(gen_data.get("summary", "*Summary not found in generated data*"))
            with col2_disp:
                st.subheader("Flashcards")
                st.json(gen_data.get("flashcards", "*Flashcards not found in generated data*"))
        else:
             st.info("Generated data is not available or is invalid for the last run.")
             st.text_area("Raw Output", current_data.get("generated_json_str", "N/A"), height=150, disabled=True)

    else:
        st.info("Run the pipeline using the sidebar to view generated output here.")

with tab_eval:
    st.header("Evaluation Results for Last Run")

    if st.session_state.current_run_index is not None:
        current_data = st.session_state.evaluation_log[st.session_state.current_run_index]
        eval_results = current_data.get("evaluation_results")

        display_evaluation_results(eval_results)

        st.divider()

        st.subheader("⭐ Provide Human Feedback (Utility)")
        st.caption("Rate how useful you found the generated summary and flashcards for *this specific run*.")

        current_log_index = st.session_state.current_run_index
        rating_key = f"rating_{current_data['timestamp']}"
        submit_key = f"submit_{current_data['timestamp']}"

        rating_already_submitted = eval_results.get("user_utility_rating") is not None

        user_rating = st.slider("Your Rating (1=Not Useful, 5=Very Useful):",
                                min_value=1, max_value=5, value=eval_results.get("user_utility_rating", 3),
                                key=rating_key,
                                disabled=rating_already_submitted)

        if st.button("Submit Rating", key=submit_key, disabled=rating_already_submitted):
            st.session_state.evaluation_log[current_log_index]["evaluation_results"]["user_utility_rating"] = user_rating
            logger.info(f"User submitted rating: {user_rating} for log index {current_log_index} (Timestamp: {current_data['timestamp']})")
            st.success(f"Rating ({user_rating}) submitted for Run ID {current_log_index + 1}!")
            st.rerun()
    else:
         st.info("Run the pipeline using the sidebar to view evaluation results here.")

with tab_log:
    st.header("📜 Run Log & History")
    st.caption("History of pipeline runs in this session.")

    if st.session_state.evaluation_log:
        log_display_data = []
        for i, entry in enumerate(reversed(st.session_state.evaluation_log)):
            eval_res = entry.get("evaluation_results", {})
            ai_res_data = eval_res.get("ai_judge_assessment", {}).get("data", {})
            acc_data = ai_res_data.get("accuracy_assessment", {}) if ai_res_data else {}
            bert_data = eval_res.get("bert_score", {})
            format_data = eval_res.get("format_check", {})
            length_data = eval_res.get("length_check", {})

            gen_error = entry.get("generator_error")
            judge_error = eval_res.get("ai_judge_assessment", {}).get("error")

            status = "Success"
            if gen_error: status = "Generator Error"
            elif not format_data.get("passed"): status = "Format Error"
            elif judge_error and "Skipped" not in judge_error: status = "AI Judge Error"

            log_item = {
                "ID": len(st.session_state.evaluation_log) - i,
                "Timestamp": time.strftime('%H:%M:%S', time.localtime(entry.get("timestamp"))),
                "Status": status,
                "Format OK?": format_data.get("passed"),
                "Length OK?": length_data.get("passed"),
                "BERTScore": f"{bert_data.get('score', 0.0):.3f}" if bert_data.get('score') is not None else "N/A",
                "BERT Pass?": bert_data.get('passed_threshold') if bert_data.get('score') is not None else "N/A",
                "AI Accuracy": ("Inaccurate" if acc_data.get("contains_inaccuracies") else "Accurate") if isinstance(acc_data.get("contains_inaccuracies"), bool) else "N/A",
                "AI Complete": ai_res_data.get("completeness_score", "N/A") if ai_res_data else "N/A",
                "AI Relevant": ai_res_data.get("relevance_score", "N/A") if ai_res_data else "N/A",
                "AI Clarity": ai_res_data.get("clarity_score", "N/A") if ai_res_data else "N/A",
                "User Rating": eval_res.get("user_utility_rating", "N/A")
            }
            log_display_data.append(log_item)

        st.dataframe(pd.DataFrame(log_display_data))

        log_ids = [item['ID'] for item in log_display_data]
        if log_ids:
            selected_id = st.selectbox("Select Log ID to view details:", options=log_ids, index=0, key="log_selector_detail")
            if selected_id:
                try:
                    selected_entry_index = len(st.session_state.evaluation_log) - selected_id
                    if 0 <= selected_entry_index < len(st.session_state.evaluation_log):
                        selected_entry = st.session_state.evaluation_log[selected_entry_index]
                        st.subheader(f"Details for Log ID: {selected_id}")
                        logger.debug(f"Displaying details for Log ID {selected_id}")
                        col_detail1, col_detail2 = st.columns(2)
                        with col_detail1:
                             with st.expander("Show Transcript"):
                                 st.text_area("Transcript", selected_entry.get("transcript"), height=200, disabled=True, key=f"detail_transcript_{selected_id}")
                        with col_detail2:
                             with st.expander("Show Generated Raw JSON"):
                                 st.text_area("Generated Raw JSON", selected_entry.get("generated_json_str", "N/A"), height=200, disabled=True, key=f"detail_rawjson_{selected_id}")

                        display_evaluation_results(selected_entry.get("evaluation_results", {}))
                    else:
                        st.error(f"Invalid selected log index: {selected_entry_index}")
                        logger.error(f"Calculated invalid log index {selected_entry_index} for selected ID {selected_id}")
                except Exception as e:
                     st.error(f"Error displaying log details: {e}")
                     logger.error(f"Error displaying log details for ID {selected_id}: {e}", exc_info=True)

        with st.expander("Show Full Raw Session Log Data (JSON)"):
            st.json(st.session_state.evaluation_log)
    else:
        st.info("Run the pipeline using the sidebar to see logs here.")

logger.info("--- Streamlit App Re-Render Complete ---")