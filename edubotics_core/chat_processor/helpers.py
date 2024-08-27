import os
from literalai import AsyncLiteralClient
from typing_extensions import TypedDict
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


async def get_thread_step_info(thread_id):
    step = await literal_client.api.get_step(thread_id)
    return step
