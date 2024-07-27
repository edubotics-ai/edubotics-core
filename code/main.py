import chainlit.data as cl_data

from modules.config.constants import (
    LLAMA_PATH,
    LITERAL_API_KEY_LOGGING,
    LITERAL_API_URL,
)
from modules.chat_processor.literal_ai import CustomLiteralDataLayer

import json
import yaml
import os
from typing import Any, Dict, no_type_check
import chainlit as cl
from modules.chat.llm_tutor import LLMTutor
from modules.chat.helpers import get_sources
import copy
from typing import Optional
from chainlit.types import ThreadDict

USER_TIMEOUT = 60_000
SYSTEM = "System 🖥️"
LLM = "LLM 🧠"
AGENT = "Agent <>"
YOU = "You 😃"
ERROR = "Error 🚫"


cl_data._data_layer = CustomLiteralDataLayer(
    api_key=LITERAL_API_KEY_LOGGING, server=LITERAL_API_URL
)


class Chatbot:
    def __init__(self):
        """
        Initialize the Chatbot class.
        """
        self.config = self._load_config()
        self.literal_client = cl_data._data_layer.client

    def _load_config(self):
        """
        Load the configuration from a YAML file.
        """
        with open("modules/config/config.yml", "r") as f:
            return yaml.safe_load(f)

    @no_type_check
    async def setup_llm(self):
        """
        Set up the LLM with the provided settings. Update the configuration and initialize the LLM tutor.
        """
        llm_settings = cl.user_session.get("llm_settings", {})
        chat_profile, retriever_method, memory_window, llm_style = (
            llm_settings.get("chat_model"),
            llm_settings.get("retriever_method"),
            llm_settings.get("memory_window"),
            llm_settings.get("llm_style"),
        )

        chain = cl.user_session.get("chain")
        print(list(chain.store.values()))
        memory_list = cl.user_session.get(
            "memory",
            (
                list(chain.store.values())[0].messages
                if len(chain.store.values()) > 0
                else []
            ),
        )
        conversation_list = []
        for message in memory_list:
            # Convert to dictionary if possible
            message_dict = message.to_dict() if hasattr(message, "to_dict") else message

            # Check if the type attribute is present as a key or attribute
            message_type = (
                message_dict.get("type", None)
                if isinstance(message_dict, dict)
                else getattr(message, "type", None)
            )

            # Check if content is present as a key or attribute
            message_content = (
                message_dict.get("content", None)
                if isinstance(message_dict, dict)
                else getattr(message, "content", None)
            )

            if message_type in ["ai", "ai_message"]:
                conversation_list.append(
                    {"type": "ai_message", "content": message_content}
                )
            elif message_type in ["human", "user_message"]:
                conversation_list.append(
                    {"type": "user_message", "content": message_content}
                )
            else:
                raise ValueError("Invalid message type")
        print("\n\n\n")
        print("history at setup_llm", conversation_list)
        print("\n\n\n")

        old_config = copy.deepcopy(self.config)
        self.config["vectorstore"]["db_option"] = retriever_method
        self.config["llm_params"]["memory_window"] = memory_window
        self.config["llm_params"]["llm_style"] = llm_style
        self.config["llm_params"]["llm_loader"] = chat_profile

        self.llm_tutor.update_llm(
            old_config, self.config
        )  # update only attributes that are changed
        self.chain = self.llm_tutor.qa_bot(memory=conversation_list)

        tags = [chat_profile, self.config["vectorstore"]["db_option"]]

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
                    id="stream_response", label="Stream response", initial=False
                ),
                cl.input_widget.Select(
                    id="llm_style",
                    label="Type of Conversation (Default Normal)",
                    values=["Normal", "ELI5", "Socratic"],
                    initial_index=0,
                ),
            ]
        ).send()

    @no_type_check
    async def inform_llm_settings(self):
        """
        Inform the user about the updated LLM settings and display them as a message.
        """
        llm_settings: Dict[str, Any] = cl.user_session.get("llm_settings", {})
        llm_tutor = cl.user_session.get("llm_tutor")
        settings_dict = {
            "model": llm_settings.get("chat_model"),
            "retriever": llm_settings.get("retriever_method"),
            "memory_window": llm_settings.get("memory_window"),
            "num_docs_in_db": (
                len(llm_tutor.vector_db)
                if llm_tutor and hasattr(llm_tutor, "vector_db")
                else 0
            ),
            "view_sources": llm_settings.get("view_sources"),
        }
        await cl.Message(
            author=SYSTEM,
            content="LLM settings have been updated. You can continue with your Query!",
            elements=[
                cl.Text(
                    name="settings",
                    display="side",
                    content=json.dumps(settings_dict, indent=4),
                    language="json",
                ),
            ],
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
        except:
            return [
                cl.Starter(
                    label="recording on CNNs?",
                    message="Where can I find the recording for the lecture on Transformers?",
                    icon="/public/adv-screen-recorder-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="where's the slides?",
                    message="When are the lectures? I can't find the schedule.",
                    icon="/public/alarmy-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="Due Date?",
                    message="When is the final project due?",
                    icon="/public/calendar-samsung-17-svgrepo-com.svg",
                ),
                cl.Starter(
                    label="Explain backprop.",
                    message="I didn't understand the math behind backprop, could you explain it?",
                    icon="/public/acastusphoton-svgrepo-com.svg",
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
        rename_dict = {"Chatbot": "AI Tutor"}
        return rename_dict.get(orig_author, orig_author)

    async def start(self):
        """
        Start the chatbot, initialize settings widgets,
        and display and load previous conversation if chat logging is enabled.
        """

        await self.make_llm_settings_widgets(self.config)
        user = cl.user_session.get("user")
        self.user = {
            "user_id": user.identifier,
            "session_id": cl.context.session.thread_id,
        }
        print(self.user)

        memory = cl.user_session.get("memory", [])

        cl.user_session.set("user", self.user)
        self.llm_tutor = LLMTutor(self.config, user=self.user)
        self.chain = self.llm_tutor.qa_bot(memory=memory)
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

        print("\n\n\n")
        print(
            "session history",
            chain.get_session_history(
                self.user["user_id"],
                self.user["session_id"],
                self.config["llm_params"]["memory_window"],
            ),
        )
        print("\n\n\n")

        llm_settings = cl.user_session.get("llm_settings", {})
        view_sources = llm_settings.get("view_sources", False)
        stream = (llm_settings.get("stream_response", True)) or (
            not self.config["llm_params"]["stream"]
        )
        user_query_dict = {"input": message.content}
        # Define the base configuration
        chain_config = {
            "configurable": {
                "user_id": self.user["user_id"],
                "conversation_id": self.user["session_id"],
                "memory_window": self.config["llm_params"]["memory_window"],
            }
        }

        stream = False

        if stream:
            res = chain.stream(user_query=user_query_dict, config=chain_config)
            res = await self.stream_response(res)
        else:
            res = await chain.invoke(user_query=user_query_dict, config=chain_config)

        answer = res.get("answer", res.get("result"))

        with cl_data._data_layer.client.step(
            type="retrieval",
            name="RAG",
            thread_id=cl.context.session.thread_id,
            # tags=self.tags,
        ) as step:
            step.input = {"question": user_query_dict["input"]}
            step.output = {
                "chat_history": res.get("chat_history"),
                "context": res.get("context"),
                "answer": answer,
                "rephrase_prompt": res.get("rephrase_prompt"),
                "qa_prompt": res.get("qa_prompt"),
            }
            step.metadata = self.config

        answer_with_sources, source_elements, sources_dict = get_sources(
            res, answer, stream=stream, view_sources=view_sources
        )

        await cl.Message(
            content=answer_with_sources, elements=source_elements, author=LLM
        ).send()

    async def on_chat_resume(self, thread: ThreadDict):
        steps = thread["steps"]
        # conversation_pairs = []
        conversation_list = []

        user_message = None
        k = self.config["llm_params"]["memory_window"]
        count = 0

        print(steps)

        for step in reversed(steps):
            print(step["type"])
            if step["name"] not in [SYSTEM]:
                if step["type"] == "user_message":
                    conversation_list.append(
                        {"type": "user_message", "content": step["output"]}
                    )
                elif step["type"] == "assistant_message":
                    if step["name"] == LLM:
                        conversation_list.append(
                            {"type": "ai_message", "content": step["output"]}
                        )
                else:
                    raise ValueError("Invalid message type")
            count += 1
            if count >= 2 * k:  # 2 * k to account for both user and assistant messages
                break

        conversation_list = conversation_list[::-1]

        print("\n\n\n")
        print("history at on_chat_resume", conversation_list)
        print(len(conversation_list))
        print("\n\n\n")
        cl.user_session.set("memory", conversation_list)
        await self.start()

    @cl.oauth_callback
    def auth_callback(
        provider_id: str,
        token: str,
        raw_user_data: Dict[str, str],
        default_user: cl.User,
    ) -> Optional[cl.User]:
        return default_user


chatbot = Chatbot()
cl.set_starters(chatbot.set_starters)
cl.author_rename(chatbot.rename)
cl.on_chat_start(chatbot.start)
cl.on_chat_resume(chatbot.on_chat_resume)
cl.on_message(chatbot.main)
cl.on_settings_update(chatbot.update_llm)
