import os
from literalai import AsyncLiteralClient
from datetime import datetime, timedelta, timezone
from modules.config.constants import COOLDOWN_TIME, TOKENS_LEFT, REGEN_TIME
from typing_extensions import TypedDict
import tiktoken
from typing import Any, Generic, List, Literal, Optional, TypeVar, Union

Field = TypeVar("Field")
Operators = TypeVar("Operators")
Value = TypeVar("Value")

BOOLEAN_OPERATORS = Literal["is", "nis"]
STRING_OPERATORS = Literal["eq", "neq", "ilike", "nilike"]
NUMBER_OPERATORS = Literal["eq", "neq", "gt", "gte", "lt", "lte"]
STRING_LIST_OPERATORS = Literal["in", "nin"]
DATETIME_OPERATORS = Literal["gte", "lte", "gt", "lt"]

OPERATORS = Union[
    BOOLEAN_OPERATORS,
    STRING_OPERATORS,
    NUMBER_OPERATORS,
    STRING_LIST_OPERATORS,
    DATETIME_OPERATORS,
]


class Filter(Generic[Field], TypedDict, total=False):
    field: Field
    operator: OPERATORS
    value: Any
    path: Optional[str]


class OrderBy(Generic[Field], TypedDict):
    column: Field
    direction: Literal["ASC", "DESC"]


threads_filterable_fields = Literal[
    "id",
    "createdAt",
    "name",
    "stepType",
    "stepName",
    "stepOutput",
    "metadata",
    "tokenCount",
    "tags",
    "participantId",
    "participantIdentifiers",
    "scoreValue",
    "duration",
]
threads_orderable_fields = Literal["createdAt", "tokenCount"]
threads_filters = List[Filter[threads_filterable_fields]]
threads_order_by = OrderBy[threads_orderable_fields]

steps_filterable_fields = Literal[
    "id",
    "name",
    "input",
    "output",
    "participantIdentifier",
    "startTime",
    "endTime",
    "metadata",
    "parentId",
    "threadId",
    "error",
    "tags",
]
steps_orderable_fields = Literal["createdAt"]
steps_filters = List[Filter[steps_filterable_fields]]
steps_order_by = OrderBy[steps_orderable_fields]

users_filterable_fields = Literal[
    "id",
    "createdAt",
    "identifier",
    "lastEngaged",
    "threadCount",
    "tokenCount",
    "metadata",
]
users_filters = List[Filter[users_filterable_fields]]

scores_filterable_fields = Literal[
    "id",
    "createdAt",
    "participant",
    "name",
    "tags",
    "value",
    "type",
    "comment",
]
scores_orderable_fields = Literal["createdAt"]
scores_filters = List[Filter[scores_filterable_fields]]
scores_order_by = OrderBy[scores_orderable_fields]

generation_filterable_fields = Literal[
    "id",
    "createdAt",
    "model",
    "duration",
    "promptLineage",
    "promptVersion",
    "tags",
    "score",
    "participant",
    "tokenCount",
    "error",
]
generation_orderable_fields = Literal[
    "createdAt",
    "tokenCount",
    "model",
    "provider",
    "participant",
    "duration",
]
generations_filters = List[Filter[generation_filterable_fields]]
generations_order_by = OrderBy[generation_orderable_fields]

literal_client = AsyncLiteralClient(api_key=os.getenv("LITERAL_API_KEY_LOGGING"))


# For consistency, use dictionary for user_info
def convert_to_dict(user_info):
    # if already a dictionary, return as is
    if isinstance(user_info, dict):
        return user_info
    if hasattr(user_info, "__dict__"):
        user_info = user_info.__dict__
    return user_info


def get_time():
    return datetime.now(timezone.utc).isoformat()


async def get_user_details(user_email_id):
    user_info = await literal_client.api.get_or_create_user(identifier=user_email_id)
    return user_info


async def update_user_info(user_info):
    # if object type, convert to dictionary
    user_info = convert_to_dict(user_info)
    await literal_client.api.update_user(
        id=user_info["id"],
        identifier=user_info["identifier"],
        metadata=user_info["metadata"],
    )


async def check_user_cooldown(user_info, current_time):

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

    # Debug: Print the cooldown end time
    print(f"Cooldown end time (ISO): {cooldown_end_time_iso}")

    # Check if the user is still in cooldown
    if elapsed_time_in_seconds < COOLDOWN_TIME:
        return True, cooldown_end_time_iso  # Return in ISO 8601 format

    user_info["metadata"]["in_cooldown"] = False
    # If not in cooldown, regenerate tokens
    await reset_tokens_for_user(user_info)

    return False, None


async def reset_tokens_for_user(user_info):
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
        regeneration_rate_per_second = max_tokens / REGEN_TIME

        # Calculate how many tokens should have been regenerated based on the elapsed time
        tokens_to_regenerate = int(
            elapsed_time_in_seconds * regeneration_rate_per_second
        )

        # Ensure the new token count does not exceed max_tokens
        new_token_count = min(current_tokens + tokens_to_regenerate, max_tokens)

        print(
            f"\n\n Adding {tokens_to_regenerate} tokens to the user, Time elapsed: {elapsed_time_in_seconds} seconds, Tokens after regeneration: {new_token_count}, Tokens before: {current_tokens} \n\n"
        )

        # Update the user's token count
        user_info["metadata"]["tokens_left"] = new_token_count

        await update_user_info(user_info)


async def get_thread_step_info(thread_id):
    step = await literal_client.api.get_step(thread_id)
    return step


def get_num_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)
