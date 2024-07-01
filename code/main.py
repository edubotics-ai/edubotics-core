import json
import textwrap
from typing import Any, Callable, Dict, List, Literal, Optional, no_type_check
import chainlit as cl
from chainlit import run_sync
from chainlit.config import config
import yaml
import os

from modules.chat.llm_tutor import LLMTutor
from modules.chat_processor.chat_processor import ChatProcessor
from modules.config.constants import LLAMA_PATH
from modules.chat.helpers import get_sources

from chainlit.input_widget import Select, Switch, Slider

USER_TIMEOUT = 60_000
SYSTEM = "System 🖥️"
LLM = "LLM 🧠"
AGENT = "Agent <>"
YOU = "You 😃"
ERROR = "Error 🚫"


class Chatbot:
    def __init__(self):
        self.llm_tutor = None
        self.chain = None
        self.chat_processor = None
        self.config = self._load_config()

    def _load_config(self):
        with open("modules/config/config.yml", "r") as f:
            config = yaml.safe_load(f)
        return config

    async def ask_helper(func, **kwargs):
        res = await func(**kwargs).send()
        while not res:
            res = await func(**kwargs).send()
        return res

    @no_type_check
    async def setup_llm(self) -> None:
        """From the session `llm_settings`, create new LLMConfig and LLM objects,
        save them in session state."""

        old_config = self.config.copy()  # create a copy of the previous config
        new_config = (
            self.config.copy()
        )  # create the new config as a copy of the previous config

        llm_settings = cl.user_session.get("llm_settings", {})
        chat_profile = llm_settings.get("chat_model")
        retriever_method = llm_settings.get("retriever_method")
        memory_window = llm_settings.get("memory_window")

        self._configure_llm(chat_profile)

        chain = cl.user_session.get("chain")
        memory = chain.memory
        new_config["vectorstore"][
            "db_option"
        ] = retriever_method  # update the retriever method in the config
        new_config["llm_params"][
            "memory_window"
        ] = memory_window  # update the memory window in the config

        self.llm_tutor.update_llm(new_config)
        self.chain = self.llm_tutor.qa_bot(memory=memory)

        tags = [chat_profile, self.config["vectorstore"]["db_option"]]
        self.chat_processor = ChatProcessor(self.llm_tutor, tags=tags)

        cl.user_session.set("chain", self.chain)
        cl.user_session.set("llm_tutor", self.llm_tutor)
        cl.user_session.set("chat_processor", self.chat_processor)

    @no_type_check
    async def update_llm(self, new_settings: Dict[str, Any]) -> None:
        """Update LLMConfig and LLM from settings, and save in session state."""
        cl.user_session.set("llm_settings", new_settings)
        await self.inform_llm_settings()
        await self.setup_llm()

    async def make_llm_settings_widgets(self, config=None):
        config = config or self.config
        await cl.ChatSettings(
            [
                cl.input_widget.Select(
                    id="chat_model",
                    label="Model Name (Default GPT-3)",
                    values=["llama", "gpt-3.5-turbo-1106", "gpt-4"],
                    initial_index=0,
                ),
                cl.input_widget.Select(
                    id="retriever_method",
                    label="Retriever (Default FAISS)",
                    values=["FAISS", "Chroma", "RAGatouille", "RAPTOR"],
                    initial_index=0,
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
                # cl.input_widget.TextInput(
                #     id="vectorstore",
                #     label="temp",
                #     initial="None",
                # ),
            ]
        ).send()  # type: ignore

    @no_type_check
    async def inform_llm_settings(self) -> None:
        llm_settings: Dict[str, Any] = cl.user_session.get("llm_settings", {})
        llm_tutor = cl.user_session.get("llm_tutor")
        settings_dict = dict(
            model=llm_settings.get("chat_model"),
            retriever=llm_settings.get("retriever_method"),
            memory_window=llm_settings.get("memory_window"),
            num_docs_in_db=len(llm_tutor.vector_db),
            view_sources=llm_settings.get("view_sources"),
        )
        await cl.Message(
            author=SYSTEM,
            content="LLM settings have been updated. You can continue with your Query!",
            elements=[
                cl.Text(
                    name="settings",
                    display="side",
                    content=json.dumps(settings_dict, indent=4),
                    language="json",
                )
            ],
        ).send()

    async def set_starters(self):
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

    async def chat_profile(self):
        return [
            cl.ChatProfile(
                name="gpt-3.5-turbo-1106",
                markdown_description="Use OpenAI API for **gpt-3.5-turbo-1106**.",
            ),
            cl.ChatProfile(
                name="gpt-4",
                markdown_description="Use OpenAI API for **gpt-4**.",
            ),
            cl.ChatProfile(
                name="Llama",
                markdown_description="Use the local LLM: **Tiny Llama**.",
            ),
        ]

    def rename(self, orig_author: str):
        rename_dict = {"Chatbot": "AI Tutor"}
        return rename_dict.get(orig_author, orig_author)

    async def start(self):
        await self.make_llm_settings_widgets(self.config)

        chat_profile = cl.user_session.get("chat_profile")
        if chat_profile:
            self._configure_llm(chat_profile)

        self.llm_tutor = LLMTutor(
            self.config, user={"user_id": "abc123", "session_id": "789"}
        )
        self.chain = self.llm_tutor.qa_bot()
        tags = [chat_profile, self.config["vectorstore"]["db_option"]]
        self.chat_processor = ChatProcessor(self.llm_tutor, tags=tags)

        cl.user_session.set("llm_tutor", self.llm_tutor)
        cl.user_session.set("chain", self.chain)
        cl.user_session.set("counter", 20)
        cl.user_session.set("chat_processor", self.chat_processor)

    async def on_chat_end(self):
        await cl.Message(content="Sorry, I have to go now. Goodbye!").send()

    async def main(self, message):
        chain = cl.user_session.get("chain")
        counter = cl.user_session.get("counter")
        llm_settings = cl.user_session.get("llm_settings", {})
        view_sources = llm_settings.get("view_sources", False)

        print("HERE")
        print(llm_settings)
        print(view_sources)
        print("\n\n")

        counter += 1
        cl.user_session.set("counter", counter)

        processor = cl.user_session.get("chat_processor")
        res = await processor.rag(message.content, chain)

        print(res)

        answer = res.get("answer", res.get("result"))

        answer_with_sources, source_elements, sources_dict = get_sources(
            res, answer, view_sources=view_sources
        )
        processor._process(message.content, answer, sources_dict)

        await cl.Message(content=answer_with_sources, elements=source_elements).send()

    def _configure_llm(self, chat_profile):
        chat_profile = chat_profile.lower()
        if chat_profile in ["gpt-3.5-turbo-1106", "gpt-4"]:
            self.config["llm_params"]["llm_loader"] = "openai"
            self.config["llm_params"]["openai_params"]["model"] = chat_profile
        elif chat_profile == "llama":
            self.config["llm_params"]["llm_loader"] = "local_llm"
            self.config["llm_params"]["local_llm_params"]["model"] = LLAMA_PATH
            self.config["llm_params"]["local_llm_params"]["model_type"] = "llama"
        elif chat_profile == "mistral":
            self.config["llm_params"]["llm_loader"] = "local_llm"
            self.config["llm_params"]["local_llm_params"]["model"] = MISTRAL_PATH
            self.config["llm_params"]["local_llm_params"]["model_type"] = "mistral"


chatbot = Chatbot()

# Register functions to Chainlit events
cl.set_starters(chatbot.set_starters)
cl.set_chat_profiles(chatbot.chat_profile)
cl.author_rename(chatbot.rename)
cl.on_chat_start(chatbot.start)
cl.on_chat_end(chatbot.on_chat_end)
cl.on_message(chatbot.main)
cl.on_settings_update(chatbot.update_llm)
