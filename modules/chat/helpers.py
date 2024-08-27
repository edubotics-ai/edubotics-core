from config.prompts import prompts
import chainlit as cl


def get_sources(res, answer, stream=True, view_sources=False):
    source_elements = []
    source_dict = {}  # Dictionary to store URL elements

    for idx, source in enumerate(res["context"]):
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


def get_prompt(config, prompt_type):
    llm_params = config["llm_params"]
    llm_loader = llm_params["llm_loader"]
    use_history = llm_params["use_history"]
    llm_style = llm_params["llm_style"].lower()

    if prompt_type == "qa":
        if llm_loader == "local_llm":
            if use_history:
                return prompts["tiny_llama"]["prompt_with_history"]
            else:
                return prompts["tiny_llama"]["prompt_no_history"]
        else:
            if use_history:
                return prompts["openai"]["prompt_with_history"][llm_style]
            else:
                return prompts["openai"]["prompt_no_history"]
    elif prompt_type == "rephrase":
        return prompts["openai"]["rephrase_prompt"]


# TODO: Do this better
def get_history_chat_resume(steps, k, SYSTEM, LLM):
    conversation_list = []
    count = 0
    for step in reversed(steps):
        if step["name"] not in [SYSTEM]:
            if step["type"] == "user_message":
                conversation_list.append(
                    {"type": "user_message", "content": step["output"]}
                )
                count += 1
            elif step["type"] == "assistant_message":
                if step["name"] == LLM:
                    conversation_list.append(
                        {"type": "ai_message", "content": step["output"]}
                    )
                    count += 1
            else:
                pass
                # raise ValueError("Invalid message type")
        # count += 1
        if count >= 2 * k:  # 2 * k to account for both user and assistant messages
            break
    conversation_list = conversation_list[::-1]
    return conversation_list


def get_history_setup_llm(memory_list):
    conversation_list = []
    i = 0
    while i < len(memory_list) - 1:
        # Process the current and next message
        current_message = memory_list[i]
        next_message = memory_list[i + 1]

        # Convert messages to dictionary if necessary
        current_message_dict = (
            current_message.to_dict()
            if hasattr(current_message, "to_dict")
            else current_message
        )
        next_message_dict = (
            next_message.to_dict() if hasattr(next_message, "to_dict") else next_message
        )

        # Check message type and content for current and next message
        current_message_type = (
            current_message_dict.get("type", None)
            if isinstance(current_message_dict, dict)
            else getattr(current_message, "type", None)
        )
        current_message_content = (
            current_message_dict.get("content", None)
            if isinstance(current_message_dict, dict)
            else getattr(current_message, "content", None)
        )

        next_message_type = (
            next_message_dict.get("type", None)
            if isinstance(next_message_dict, dict)
            else getattr(next_message, "type", None)
        )
        next_message_content = (
            next_message_dict.get("content", None)
            if isinstance(next_message_dict, dict)
            else getattr(next_message, "content", None)
        )

        # Check if the current message is user message and the next one is AI message
        if current_message_type in ["human", "user_message"] and next_message_type in [
            "ai",
            "ai_message",
        ]:
            conversation_list.append(
                {"type": "user_message", "content": current_message_content}
            )
            conversation_list.append(
                {"type": "ai_message", "content": next_message_content}
            )
            i += 2  # Skip the next message since it has been paired
        else:
            i += 1  # Move to the next message if not a valid pair (example user message, followed by the cooldown system message)

    return conversation_list


def get_last_config(steps):
    # TODO: Implement this function
    return None
