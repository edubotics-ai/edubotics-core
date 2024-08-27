import chainlit.data as cl_data
import asyncio
from config.constants import (
    LITERAL_API_KEY_LOGGING,
    LITERAL_API_URL,
)
from edubotics_core.chat_processor.literal_ai import CustomLiteralDataLayer
import json
from typing import Any, Dict, no_type_check
import chainlit as cl
from edubotics_core.chat.llm_tutor import LLMTutor
from edubotics_core.chat.helpers import (
    get_sources,
    get_history_chat_resume,
    get_history_setup_llm,
    # get_last_config,
)
from edubotics_core.chat_processor.helpers import (
    update_user_info,
    get_user_details,
)
from helpers import (
    check_user_cooldown,
    reset_tokens_for_user,
)
from helpers import get_time
import copy
from typing import Optional
from chainlit.types import ThreadDict
import base64
from langchain_community.callbacks import get_openai_callback
from datetime import datetime, timezone
from config.config_manager import config_manager

USER_TIMEOUT = 60_000
SYSTEM = "System"
LLM = "AI Tutor"
AGENT = "Agent"
YOU = "User"
ERROR = "Error"

# set config
config = config_manager.get_config().dict()


async def setup_data_layer():
    """
    Set up the data layer for chat logging.
    """
    if config["chat_logging"]["log_chat"]:
        data_layer = CustomLiteralDataLayer(
            api_key=LITERAL_API_KEY_LOGGING, server=LITERAL_API_URL
        )
    else:
        data_layer = None

    return data_layer


async def update_user_from_chainlit(user, token_count=0):
    if "admin" not in user.metadata["role"]:
        user.metadata["tokens_left"] = user.metadata["tokens_left"] - token_count
        user.metadata["all_time_tokens_allocated"] = (
            user.metadata["all_time_tokens_allocated"] - token_count
        )
        user.metadata["tokens_left_at_last_message"] = user.metadata[
            "tokens_left"
        ]  # tokens_left will keep regenerating outside of chainlit
    user.metadata["last_message_time"] = get_time()
    await update_user_info(user)

    tokens_left = user.metadata["tokens_left"]
    if tokens_left < 0:
        tokens_left = 0
    return tokens_left


class Chatbot:
    def __init__(self, config):
        """
        Initialize the Chatbot class.
        """
        self.config = config

    @no_type_check
    async def setup_llm(self):
        """
        Set up the LLM with the provided settings. Update the configuration and initialize the LLM tutor.

        #TODO: Clean this up.
        """

        llm_settings = cl.user_session.get("llm_settings", {})
        (
            chat_profile,
            retriever_method,
            memory_window,
            llm_style,
            generate_follow_up,
            chunking_mode,
        ) = (
            llm_settings.get("chat_model"),
            llm_settings.get("retriever_method"),
            llm_settings.get("memory_window"),
            llm_settings.get("llm_style"),
            llm_settings.get("follow_up_questions"),
            llm_settings.get("chunking_mode"),
        )

        chain = cl.user_session.get("chain")
        memory_list = cl.user_session.get(
            "memory",
            (
                list(chain.store.values())[0].messages
                if len(chain.store.values()) > 0
                else []
            ),
        )
        conversation_list = get_history_setup_llm(memory_list)

        old_config = copy.deepcopy(self.config)
        self.config["vectorstore"]["db_option"] = retriever_method
        self.config["llm_params"]["memory_window"] = memory_window
        self.config["llm_params"]["llm_style"] = llm_style
        self.config["llm_params"]["llm_loader"] = chat_profile
        self.config["llm_params"]["generate_follow_up"] = generate_follow_up
        self.config["splitter_options"]["chunking_mode"] = chunking_mode

        self.llm_tutor.update_llm(
            old_config, self.config
        )  # update only llm attributes that are changed
        self.chain = self.llm_tutor.qa_bot(
            memory=conversation_list,
        )

        cl.user_session.set("chain", self.chain)
        cl.user_session.set("llm_tutor", self.llm_tutor)

    @no_type_check
    async def update_llm(self, new_settings: Dict[str, Any]):
        """
        Update the LLM settings and reinitialize the LLM with the new settings.

        Args:
            new_settings (Dict[str, Any]): The new settings to update.
        """
        cl.user_session.set("llm_settings", new_settings)
        await self.inform_llm_settings()
        await self.setup_llm()

    async def make_llm_settings_widgets(self, config=None):
        """
        Create and send the widgets for LLM settings configuration.

        Args:
            config: The configuration to use for setting up the widgets.
        """
        config = config or self.config
        await cl.ChatSettings(
            [
                cl.input_widget.Select(
                    id="chat_model",
                    label="Model Name (Default GPT-3)",
                    values=["local_llm", "gpt-3.5-turbo-1106", "gpt-4", "gpt-4o-mini"],
                    initial_index=[
                        "local_llm",
                        "gpt-3.5-turbo-1106",
                        "gpt-4",
                        "gpt-4o-mini",
                    ].index(config["llm_params"]["llm_loader"]),
                ),
                cl.input_widget.Select(
                    id="retriever_method",
                    label="Retriever (Default FAISS)",
                    values=["FAISS", "Chroma", "RAGatouille", "RAPTOR"],
                    initial_index=["FAISS", "Chroma", "RAGatouille", "RAPTOR"].index(
                        config["vectorstore"]["db_option"]
                    ),
                ),
                cl.input_widget.Slider(
                    id="memory_window",
                    label="Memory Window (Default 3)",
                    initial=3,
                    min=0,
                    max=10,
                    step=1,
                ),
                cl.input_widget.Switch(
                    id="view_sources", label="View Sources", initial=False
                ),
                cl.input_widget.Switch(
                    id="stream_response",
                    label="Stream response",
                    initial=config["llm_params"]["stream"],
                ),
                cl.input_widget.Select(
                    id="chunking_mode",
                    label="Chunking mode",
                    values=["fixed", "semantic"],
                    initial_index=1,
                ),
                cl.input_widget.Switch(
                    id="follow_up_questions",
                    label="Generate follow up questions",
                    initial=False,
                ),
                cl.input_widget.Select(
                    id="llm_style",
                    label="Type of Conversation (Default Normal)",
                    values=["Normal", "ELI5"],
                    initial_index=0,
                ),
            ]
        ).send()

    @no_type_check
    async def inform_llm_settings(self):
        """
        Inform the user about the updated LLM settings and display them as a message.
        """
        await cl.Message(
            author=SYSTEM,
            content="LLM settings have been updated. You can continue with your Query!",
        ).send()

    async def set_starters(self):
        """
        Set starter messages for the chatbot.
        """
        # Return Starters only if the chat is new

        try:
            thread = cl_data._data_layer.get_thread(
                cl.context.session.thread_id
            )  # see if the thread has any steps
            if thread.steps or len(thread.steps) > 0:
                return None
        except Exception as e:
            print(e)
            return [
                cl.Starter(
                    label="recording on Transformers?",
                    message="Where can I find the recording for the lecture on Transformers?",
                    icon="/public/assets/images/starter_icons/adv-screen-recorder-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="where's the schedule?",
                    message="When are the lectures? I can't find the schedule.",
                    icon="/public/assets/images/starter_icons/alarmy-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="Due Date?",
                    message="When is the final project due?",
                    icon="/public/assets/images/starter_icons/calendar-samsung-17-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="Explain backprop.",
                    message="I didn't understand the math behind backprop, could you explain it?",
                    icon="/public/assets/images/starter_icons/acastusphoton-svgrepo-com.svg",
                ),
            ]

    def rename(self, orig_author: str):
        """
        Rename the original author to a more user-friendly name.

        Args:
            orig_author (str): The original author's name.

        Returns:
            str: The renamed author.
        """
        rename_dict = {"Chatbot": LLM}
        return rename_dict.get(orig_author, orig_author)

    async def start(self):
        """
        Start the chatbot, initialize settings widgets,
        and display and load previous conversation if chat logging is enabled.
        """

        await self.make_llm_settings_widgets(self.config)  # Reload the settings widgets

        user = cl.user_session.get("user")

        # TODO: remove self.user with cl.user_session.get("user")
        try:
            self.user = {
                "user_id": user.identifier,
                "session_id": cl.context.session.thread_id,
            }
        except Exception as e:
            print(e)
            self.user = {
                "user_id": "guest",
                "session_id": cl.context.session.thread_id,
            }

        memory = cl.user_session.get("memory", [])
        self.llm_tutor = LLMTutor(self.config, user=self.user)

        self.chain = self.llm_tutor.qa_bot(
            memory=memory,
        )
        self.question_generator = self.llm_tutor.question_generator
        cl.user_session.set("llm_tutor", self.llm_tutor)
        cl.user_session.set("chain", self.chain)

    async def stream_response(self, response):
        """
        Stream the response from the LLM.

        Args:
            response: The response from the LLM.
        """
        msg = cl.Message(content="")
        await msg.send()

        output = {}
        for chunk in response:
            if "answer" in chunk:
                await msg.stream_token(chunk["answer"])

            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]
        return output

    async def main(self, message):
        """
        Process and Display the Conversation.

        Args:
            message: The incoming chat message.
        """

        chain = cl.user_session.get("chain")
        token_count = 0  # initialize token count
        if not chain:
            await self.start()  # start the chatbot if the chain is not present
            chain = cl.user_session.get("chain")

        # update user info with last message time
        user = cl.user_session.get("user")
        await reset_tokens_for_user(
            user,
            self.config["token_config"]["tokens_left"],
            self.config["token_config"]["regen_time"],
        )
        updated_user = await get_user_details(user.identifier)
        user.metadata = updated_user.metadata
        cl.user_session.set("user", user)

        # see if user has token credits left
        # if not, return message saying they have run out of tokens
        if user.metadata["tokens_left"] <= 0 and "admin" not in user.metadata["role"]:
            current_datetime = get_time()
            cooldown, cooldown_end_time = await check_user_cooldown(
                user,
                current_datetime,
                self.config["token_config"]["cooldown_time"],
                self.config["token_config"]["tokens_left"],
                self.config["token_config"]["regen_time"],
            )
            if cooldown:
                # get time left in cooldown
                # convert both to datetime objects
                cooldown_end_time = datetime.fromisoformat(cooldown_end_time).replace(
                    tzinfo=timezone.utc
                )
                current_datetime = datetime.fromisoformat(current_datetime).replace(
                    tzinfo=timezone.utc
                )
                cooldown_time_left = cooldown_end_time - current_datetime
                # Get the total seconds
                total_seconds = int(cooldown_time_left.total_seconds())
                # Calculate hours, minutes, and seconds
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                # Format the time as 00 hrs 00 mins 00 secs
                formatted_time = f"{hours:02} hrs {minutes:02} mins {seconds:02} secs"
                await cl.Message(
                    content=(
                        "Ah, seems like you have run out of tokens...Click "
                        '<a href="/cooldown" style="color: #0000CD; text-decoration: none;" target="_self">here</a> for more info. Please come back after {}'.format(
                            formatted_time
                        )
                    ),
                    author=SYSTEM,
                ).send()
                user.metadata["in_cooldown"] = True
                await update_user_info(user)
                return
            else:
                await cl.Message(
                    content=(
                        "Ah, seems like you don't have any tokens left...Please wait while we regenerate your tokens. Click "
                        '<a href="/cooldown" style="color: #0000CD; text-decoration: none;" target="_self">here</a> to view your token credits.'
                    ),
                    author=SYSTEM,
                ).send()
                return

        user.metadata["in_cooldown"] = False

        llm_settings = cl.user_session.get("llm_settings", {})
        view_sources = llm_settings.get("view_sources", False)
        stream = llm_settings.get("stream_response", False)
        stream = False  # Fix streaming
        user_query_dict = {"input": message.content}
        # Define the base configuration
        cb = cl.AsyncLangchainCallbackHandler()
        chain_config = {
            "configurable": {
                "user_id": self.user["user_id"],
                "conversation_id": self.user["session_id"],
                "memory_window": self.config["llm_params"]["memory_window"],
            },
            "callbacks": (
                [cb]
                if cl_data._data_layer and self.config["chat_logging"]["callbacks"]
                else None
            ),
        }

        with get_openai_callback() as token_count_cb:
            if stream:
                res = chain.stream(user_query=user_query_dict, config=chain_config)
                res = await self.stream_response(res)
            else:
                res = await chain.invoke(
                    user_query=user_query_dict,
                    config=chain_config,
                )
        token_count += token_count_cb.total_tokens

        answer = res.get("answer", res.get("result"))

        answer_with_sources, source_elements, sources_dict = get_sources(
            res, answer, stream=stream, view_sources=view_sources
        )
        answer_with_sources = answer_with_sources.replace("$$", "$")

        actions = []

        if self.config["llm_params"]["generate_follow_up"]:
            cb_follow_up = cl.AsyncLangchainCallbackHandler()
            config = {
                "callbacks": (
                    [cb_follow_up]
                    if cl_data._data_layer and self.config["chat_logging"]["callbacks"]
                    else None
                )
            }
            with get_openai_callback() as token_count_cb:
                list_of_questions = await self.question_generator.generate_questions(
                    query=user_query_dict["input"],
                    response=answer,
                    chat_history=res.get("chat_history"),
                    context=res.get("context"),
                    config=config,
                )

            token_count += token_count_cb.total_tokens

            for question in list_of_questions:
                actions.append(
                    cl.Action(
                        name="follow up question",
                        value="example_value",
                        description=question,
                        label=question,
                    )
                )

        # # update user info with token count
        tokens_left = await update_user_from_chainlit(user, token_count)

        answer_with_sources += (
            '\n\n<footer><span style="font-size: 0.8em; text-align: right; display: block;">Tokens Left: '
            + str(tokens_left)
            + "</span></footer>\n"
        )

        await cl.Message(
            content=answer_with_sources,
            elements=source_elements,
            author=LLM,
            actions=actions,
        ).send()

    async def on_chat_resume(self, thread: ThreadDict):
        # thread_config = None
        steps = thread["steps"]
        k = self.config["llm_params"][
            "memory_window"
        ]  # on resume, alwyas use the default memory window
        conversation_list = get_history_chat_resume(steps, k, SYSTEM, LLM)
        # thread_config = get_last_config(
        #     steps
        # )  # TODO: Returns None for now - which causes config to be reloaded with default values
        cl.user_session.set("memory", conversation_list)
        await self.start()

    @cl.header_auth_callback
    def header_auth_callback(headers: dict) -> Optional[cl.User]:
        # try: # TODO: Add try-except block after testing
        # TODO: Implement to get the user information from the headers (not the cookie)
        cookie = headers.get("cookie")  # gets back a str
        # Create a dictionary from the pairs
        cookie_dict = {}
        for pair in cookie.split("; "):
            key, value = pair.split("=", 1)
            # Strip surrounding quotes if present
            cookie_dict[key] = value.strip('"')

        decoded_user_info = base64.b64decode(
            cookie_dict.get("X-User-Info", "")
        ).decode()
        decoded_user_info = json.loads(decoded_user_info)

        return cl.User(
            id=decoded_user_info["literalai_info"]["id"],
            identifier=decoded_user_info["literalai_info"]["identifier"],
            metadata=decoded_user_info["literalai_info"]["metadata"],
        )

    async def on_follow_up(self, action: cl.Action):
        user = cl.user_session.get("user")
        message = await cl.Message(
            content=action.description,
            type="user_message",
            author=user.identifier,
        ).send()
        async with cl.Step(
            name="on_follow_up", type="run", parent_id=message.id
        ) as step:
            await self.main(message)
            step.output = message.content


chatbot = Chatbot(config=config)


async def start_app():
    cl_data._data_layer = await setup_data_layer()
    chatbot.literal_client = cl_data._data_layer.client if cl_data._data_layer else None
    cl.set_starters(chatbot.set_starters)
    cl.author_rename(chatbot.rename)
    cl.on_chat_start(chatbot.start)
    cl.on_chat_resume(chatbot.on_chat_resume)
    cl.on_message(chatbot.main)
    cl.on_settings_update(chatbot.update_llm)
    cl.action_callback("follow up question")(chatbot.on_follow_up)


loop = asyncio.get_event_loop()
if loop.is_running():
    asyncio.ensure_future(start_app())
else:
    asyncio.run(start_app())
