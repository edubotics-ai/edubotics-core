def get_prompt(config, prompt_type, all_prompts):
    llm_params = config["llm_params"]
    llm_loader = llm_params["llm_loader"]
    use_history = llm_params["use_history"]
    llm_style = llm_params.get("llm_style", "normal").lower()

    print(all_prompts.keys())

    # Validate llm_loader
    if llm_loader not in all_prompts:
        raise ValueError(f"Invalid llm_loader: {llm_loader}")

    loader_prompts = all_prompts[llm_loader]

    # Determine the appropriate prompt key
    if use_history:
        history_key = "prompt_with_history"
    else:
        history_key = "prompt_no_history"

    # Handle the case where the prompt type is a specific one like 'rephrase_prompt'
    if prompt_type in loader_prompts:
        selected_prompt = loader_prompts[prompt_type]
    # Handle the case where the prompt type is a generic one like 'prompt_with_history' or 'prompt_no_history'
    elif history_key in loader_prompts:
        selected_prompt = loader_prompts[history_key]
    else:
        raise ValueError(
            f"No valid prompt found for {prompt_type} or {history_key} in {llm_loader}"
        )

    # If selected_prompt is a dictionary (e.g., different styles), return the appropriate style
    if isinstance(selected_prompt, dict):
        if llm_style not in selected_prompt:
            raise ValueError(f"Invalid llm_style: {llm_style} for {llm_loader}")
        return selected_prompt[llm_style]
    else:
        return selected_prompt


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
