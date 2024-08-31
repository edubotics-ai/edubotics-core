from datetime import datetime, timedelta, timezone
import tiktoken
from edubotics_core.chat_processor.helpers import update_user_info, convert_to_dict


def get_sources(res, answer, stream=True, view_sources=False):
    source_elements = []
    source_dict = {}  # Dictionary to store URL elements

    for idx, source in enumerate(res["context"]):
        print(source)
        source_metadata = source.metadata
        url = source_metadata.get("source", "N/A")
        score = source_metadata.get("score", "N/A")
        page = source_metadata.get("page", 1)

        lecture_tldr = source_metadata.get("tldr", "N/A")
        lecture_recording = source_metadata.get("lecture_recording", "N/A")
        suggested_readings = source_metadata.get("suggested_readings", "N/A")
        date = source_metadata.get("date", "N/A")

        source_type = source_metadata.get("source_type", "N/A")

        url_name = f"{url}_{page}"
        if url_name not in source_dict:
            source_dict[url_name] = {
                "text": source.page_content,
                "url": url,
                "score": score,
                "page": page,
                "lecture_tldr": lecture_tldr,
                "lecture_recording": lecture_recording,
                "suggested_readings": suggested_readings,
                "date": date,
                "source_type": source_type,
            }
        else:
            source_dict[url_name]["text"] += f"\n\n{source.page_content}"

    full_answer = ""  # Not to include the answer again if streaming

    if not stream:  # First, display the answer if not streaming
        # full_answer = "**Answer:**\n"
        full_answer += answer

    if view_sources:
        # Then, display the sources
        # check if the answer has sources
        if len(source_dict) == 0:
            full_answer += "\n\n**No sources found.**"
            return full_answer, source_elements, source_dict
        else:
            full_answer += "\n\n**Sources:**\n"
            for idx, (url_name, source_data) in enumerate(source_dict.items()):
                full_answer += f"\nSource {idx + 1} (Score: {source_data['score']}): {source_data['url']}\n"

                name = f"Source {idx + 1} Text\n"
                full_answer += name
                source_elements.append(
                    cl.Text(name=name, content=source_data["text"], display="side")
                )

                # Add a PDF element if the source is a PDF file
                if source_data["url"].lower().endswith(".pdf"):
                    name = f"Source {idx + 1} PDF\n"
                    full_answer += name
                    pdf_url = f"{source_data['url']}#page={source_data['page']+1}"
                    source_elements.append(
                        cl.Pdf(name=name, url=pdf_url, display="side")
                    )

            full_answer += "\n**Metadata:**\n"
            for idx, (url_name, source_data) in enumerate(source_dict.items()):
                full_answer += f"\nSource {idx + 1} Metadata:\n"
                source_elements.append(
                    cl.Text(
                        name=f"Source {idx + 1} Metadata",
                        content=f"Source: {source_data['url']}\n"
                        f"Page: {source_data['page']}\n"
                        f"Type: {source_data['source_type']}\n"
                        f"Date: {source_data['date']}\n"
                        f"TL;DR: {source_data['lecture_tldr']}\n"
                        f"Lecture Recording: {source_data['lecture_recording']}\n"
                        f"Suggested Readings: {source_data['suggested_readings']}\n",
                        display="side",
                    )
                )

    return full_answer, source_elements, source_dict


def get_time():
    return datetime.now(timezone.utc).isoformat()


async def check_user_cooldown(
    user_info, current_time, COOLDOWN_TIME, TOKENS_LEFT, REGEN_TIME
):
    # # Check if no tokens left
    tokens_left = user_info.metadata.get("tokens_left", 0)
    if tokens_left > 0 and not user_info.metadata.get("in_cooldown", False):
        return False, None

    user_info = convert_to_dict(user_info)
    last_message_time_str = user_info["metadata"].get("last_message_time")

    # Convert from ISO format string to datetime object and ensure UTC timezone
    last_message_time = datetime.fromisoformat(last_message_time_str).replace(
        tzinfo=timezone.utc
    )
    current_time = datetime.fromisoformat(current_time).replace(tzinfo=timezone.utc)

    # Calculate the elapsed time
    elapsed_time = current_time - last_message_time
    elapsed_time_in_seconds = elapsed_time.total_seconds()

    # Calculate when the cooldown period ends
    cooldown_end_time = last_message_time + timedelta(seconds=COOLDOWN_TIME)
    cooldown_end_time_iso = cooldown_end_time.isoformat()

    # Check if the user is still in cooldown
    if elapsed_time_in_seconds < COOLDOWN_TIME:
        return True, cooldown_end_time_iso  # Return in ISO 8601 format

    user_info["metadata"]["in_cooldown"] = False
    # If not in cooldown, regenerate tokens
    await reset_tokens_for_user(user_info, TOKENS_LEFT, REGEN_TIME)

    return False, None


async def reset_tokens_for_user(user_info, TOKENS_LEFT, REGEN_TIME):
    user_info = convert_to_dict(user_info)
    last_message_time_str = user_info["metadata"].get("last_message_time")

    last_message_time = datetime.fromisoformat(last_message_time_str).replace(
        tzinfo=timezone.utc
    )
    current_time = datetime.fromisoformat(get_time()).replace(tzinfo=timezone.utc)

    # Calculate the elapsed time since the last message
    elapsed_time_in_seconds = (current_time - last_message_time).total_seconds()

    # Current token count (can be negative)
    current_tokens = user_info["metadata"].get("tokens_left_at_last_message", 0)
    current_tokens = min(current_tokens, TOKENS_LEFT)

    # Maximum tokens that can be regenerated
    max_tokens = user_info["metadata"].get("max_tokens", TOKENS_LEFT)

    # Calculate how many tokens should have been regenerated proportionally
    if current_tokens < max_tokens:
        # Calculate the regeneration rate per second based on REGEN_TIME for full regeneration
        # If current_tokens is close to 0, then the regeneration rate is relatively high, and if current_tokens is close to max_tokens, then the regeneration rate is relatively low
        regeneration_rate_per_second = (
            max_tokens - max(current_tokens, 0)
        ) / REGEN_TIME

        # Calculate how many tokens should have been regenerated based on the elapsed time
        tokens_to_regenerate = int(
            elapsed_time_in_seconds * regeneration_rate_per_second
        )

        # Ensure the new token count does not exceed max_tokens
        new_token_count = min(current_tokens + tokens_to_regenerate, max_tokens)

        # Update the user's token count
        user_info["metadata"]["tokens_left"] = new_token_count

        await update_user_info(user_info)


def get_num_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)
