from edubotics_core.chat.helpers import get_prompt
from edubotics_core.chat.chat_model_loader import ChatModelLoader
from edubotics_core.vectorstore.store_manager import VectorStoreManager
from edubotics_core.retriever.retriever import Retriever
from edubotics_core.chat.langchain.langchain_rag import (
    Langchain_RAG_V2,
    QuestionGenerator,
)


class LLMTutor:
    def __init__(self, config, user, logger=None):
        """
        Initialize the LLMTutor class.

        Args:
            config (dict): Configuration dictionary.
            user (str): User identifier.
            logger (Logger, optional): Logger instance. Defaults to None.
        """
        self.config = config
        self.llm = self.load_llm()
        self.user = user
        self.logger = logger
        self.vector_db = VectorStoreManager(config, logger=self.logger).load_database()
        self.qa_prompt = get_prompt(config, "qa")  # Initialize qa_prompt
        self.rephrase_prompt = get_prompt(
            config, "rephrase"
        )  # Initialize rephrase_prompt

        # TODO: Removed this functionality for now, don't know if we need it
        # if self.config["vectorstore"]["embedd_files"]:
        #     self.vector_db.create_database()
        #     self.vector_db.save_database()

    def update_llm(self, old_config, new_config):
        """
        Update the LLM and VectorStoreManager based on new configuration.

        Args:
            new_config (dict): New configuration dictionary.
        """
        changes = self.get_config_changes(old_config, new_config)

        if "llm_params.llm_loader" in changes:
            self.llm = self.load_llm()  # Reinitialize LLM if chat_model changes

        if "vectorstore.db_option" in changes:
            self.vector_db = VectorStoreManager(
                self.config, logger=self.logger
            ).load_database()  # Reinitialize VectorStoreManager if vectorstore changes

            # TODO: Removed this functionality for now, don't know if we need it
            # if self.config["vectorstore"]["embedd_files"]:
            #     self.vector_db.create_database()
            #     self.vector_db.save_database()

        if "llm_params.llm_style" in changes:
            self.qa_prompt = get_prompt(
                self.config, "qa"
            )  # Update qa_prompt if ELI5 changes

    def get_config_changes(self, old_config, new_config):
        """
        Get the changes between the old and new configuration.

        Args:
            old_config (dict): Old configuration dictionary.
            new_config (dict): New configuration dictionary.

        Returns:
            dict: Dictionary containing the changes.
        """
        changes = {}

        def compare_dicts(old, new, parent_key=""):
            for key in new:
                full_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(new[key], dict) and isinstance(old.get(key), dict):
                    compare_dicts(old.get(key, {}), new[key], full_key)
                elif old.get(key) != new[key]:
                    changes[full_key] = (old.get(key), new[key])
            # Include keys that are in old but not in new
            for key in old:
                if key not in new:
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    changes[full_key] = (old[key], None)

        compare_dicts(old_config, new_config)
        return changes

    def retrieval_qa_chain(
        self, llm, qa_prompt, rephrase_prompt, db, memory=None, callbacks=None
    ):
        """
        Create a Retrieval QA Chain.

        Args:
            llm (LLM): The language model instance.
            qa_prompt (str): The QA prompt string.
            rephrase_prompt (str): The rephrase prompt string.
            db (VectorStore): The vector store instance.
            memory (Memory, optional): Memory instance. Defaults to None.

        Returns:
            Chain: The retrieval QA chain instance.
        """
        retriever = Retriever(self.config)._return_retriever(db)

        if self.config["llm_params"]["llm_arch"] == "langchain":
            self.qa_chain = Langchain_RAG_V2(
                llm=llm,
                memory=memory,
                retriever=retriever,
                qa_prompt=qa_prompt,
                rephrase_prompt=rephrase_prompt,
                config=self.config,
                callbacks=callbacks,
            )

            self.question_generator = QuestionGenerator()
        else:
            raise ValueError(
                f"Invalid LLM Architecture: {self.config['llm_params']['llm_arch']}"
            )
        return self.qa_chain

    def load_llm(self):
        """
        Load the language model.

        Returns:
            LLM: The loaded language model instance.
        """
        chat_model_loader = ChatModelLoader(self.config)
        llm = chat_model_loader.load_chat_model()
        return llm

    def qa_bot(self, memory=None, callbacks=None):
        """
        Create a QA bot instance.

        Args:
            memory (Memory, optional): Memory instance. Defaults to None.
            qa_prompt (str, optional): QA prompt string. Defaults to None.
            rephrase_prompt (str, optional): Rephrase prompt string. Defaults to None.

        Returns:
            Chain: The QA bot chain instance.
        """
        # sanity check to see if there are any documents in the database
        if len(self.vector_db) == 0:
            raise ValueError(
                "No documents in the database. Populate the database first."
            )

        qa = self.retrieval_qa_chain(
            self.llm,
            self.qa_prompt,
            self.rephrase_prompt,
            self.vector_db,
            memory,
            callbacks=callbacks,
        )

        return qa
