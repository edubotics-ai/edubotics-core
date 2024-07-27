from modules.config.prompts import prompts
import chainlit as cl


def get_sources(res, answer, stream=True, view_sources=False):
    source_elements = []
    source_dict = {}  # Dictionary to store URL elements

    print("\n\n\n")
    print(res["context"])
    print(len(res["context"]))
    print("\n\n\n")

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
        print("url")
        print(url_name)
        print("\n\n\n")
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
        full_answer = "**Answer:**\n"
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
