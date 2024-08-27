from datetime import datetime, timedelta, timezone
import tiktoken
from edubotics_core.chat_processor.helpers import update_user_info, convert_to_dict


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
