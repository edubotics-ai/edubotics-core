import json
import yaml
import os
from typing import Any, Dict, no_type_check
import chainlit as cl
from modules.chat.llm_tutor import LLMTutor
from modules.chat_processor.chat_processor import ChatProcessor
from modules.config.constants import LLAMA_PATH
from modules.chat.helpers import get_sources
import copy
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

USER_TIMEOUT = 60_000
SYSTEM = "System 🖥️"
LLM = "LLM 🧠"
AGENT = "Agent <>"
YOU = "You 😃"
ERROR = "Error 🚫"


class Chatbot:
    def __init__(self):
        """
        Initialize the Chatbot class.
        """
        self.config = self._load_config()

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
        memory = chain.memory if chain else []

        old_config = copy.deepcopy(self.config)
        self.config["vectorstore"]["db_option"] = retriever_method
        self.config["llm_params"]["memory_window"] = memory_window
        self.config["llm_params"]["llm_style"] = llm_style
        self.config["llm_params"]["llm_loader"] = chat_profile

        self.llm_tutor.update_llm(
            old_config, self.config
        )  # update only attributes that are changed
        self.chain = self.llm_tutor.qa_bot(memory=memory)

        tags = [chat_profile, self.config["vectorstore"]["db_option"]]
        self.chat_processor.config = self.config

        cl.user_session.set("chain", self.chain)
        cl.user_session.set("llm_tutor", self.llm_tutor)
        cl.user_session.set("chat_processor", self.chat_processor)

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
                    values=["local_llm", "gpt-3.5-turbo-1106", "gpt-4"],
                    initial_index=["local_llm", "gpt-3.5-turbo-1106", "gpt-4"].index(config["llm_params"]["llm_loader"]),
                ),
                cl.input_widget.Select(
                    id="retriever_method",
                    label="Retriever (Default FAISS)",
                    values=["FAISS", "Chroma", "RAGatouille", "RAPTOR"],
                    initial_index=["FAISS", "Chroma", "RAGatouille", "RAPTOR"].index(config["vectorstore"]["db_option"])
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
        await cl.Message(content="Welcome back! Setting up your session...").send()

        await self.make_llm_settings_widgets(self.config)
        user = cl.user_session.get("user")
        self.user = {
            "user_id": user.identifier,
            "session_id": "1234",
        }
        cl.user_session.set("user", self.user)
        self.chat_processor = ChatProcessor(self.config, self.user)
        self.llm_tutor = LLMTutor(self.config, user=self.user)
        if self.config["chat_logging"]["log_chat"]:
            # get previous conversation of the user
            memory = self.chat_processor.processor.prev_conv
            if len(self.chat_processor.processor.prev_conv) > 0:
                for idx, conv in enumerate(self.chat_processor.processor.prev_conv):
                    await cl.Message(
                        author="User", content=conv[0], type="user_message"
                    ).send()
                    await cl.Message(author="AI Tutor", content=conv[1]).send()
        else:
            memory = []
        self.chain = self.llm_tutor.qa_bot(memory=memory)
        cl.user_session.set("llm_tutor", self.llm_tutor)
        cl.user_session.set("chain", self.chain)
        cl.user_session.set("chat_processor", self.chat_processor)

    async def on_chat_end(self):
        """
        Handle the end of the chat session by sending a goodbye message.
        # TODO: Not used as of now - useful when the implementation for the conversation limiting is implemented
        """
        await cl.Message(content="Sorry, I have to go now. Goodbye!").send()

    async def main(self, message):
        """
        Process and Display the Conversation.

        Args:
            message: The incoming chat message.
        """
        chain = cl.user_session.get("chain")
        llm_settings = cl.user_session.get("llm_settings", {})
        view_sources = llm_settings.get("view_sources", False)

        processor = cl.user_session.get("chat_processor")
        res = await processor.rag(message.content, chain)

        # TODO: STREAM MESSAGE
        msg = cl.Message(content="")
        await msg.send()

        output = {}
        for chunk in res:
            if 'answer' in chunk:
                await msg.stream_token(chunk['answer'])

            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]

        answer = output.get("answer", output.get("result"))

        answer_with_sources, source_elements, sources_dict = get_sources(
            output, answer, view_sources=view_sources
        )
        processor._process(message.content, answer, sources_dict)

        await cl.Message(content=answer_with_sources, elements=source_elements).send()

    def auth_callback(self, username: str, password: str) -> Optional[cl.User]:
            return cl.User(
                identifier=username,
                metadata={"role": "admin", "provider": "credentials"},
            )

chatbot = Chatbot()
cl.password_auth_callback(chatbot.auth_callback)
cl.set_starters(chatbot.set_starters)
cl.author_rename(chatbot.rename)
cl.on_chat_start(chatbot.start)
cl.on_chat_end(chatbot.on_chat_end)
cl.on_message(chatbot.main)
cl.on_settings_update(chatbot.update_llm)
