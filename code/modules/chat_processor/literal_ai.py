from chainlit.data import ChainlitDataLayer, queue_until_user_message


# update custom methods here (Ref: https://github.com/Chainlit/chainlit/blob/4b533cd53173bcc24abe4341a7108f0070d60099/backend/chainlit/data/__init__.py)
class CustomLiteralDataLayer(ChainlitDataLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @queue_until_user_message()
    async def create_step(self, step_dict: "StepDict"):
        metadata = dict(
            step_dict.get("metadata", {}),
            **{
                "waitForAnswer": step_dict.get("waitForAnswer"),
                "language": step_dict.get("language"),
                "showInput": step_dict.get("showInput"),
            },
        )

        step: LiteralStepDict = {
            "createdAt": step_dict.get("createdAt"),
            "startTime": step_dict.get("start"),
            "endTime": step_dict.get("end"),
            "generation": step_dict.get("generation"),
            "id": step_dict.get("id"),
            "parentId": step_dict.get("parentId"),
            "name": step_dict.get("name"),
            "threadId": step_dict.get("threadId"),
            "type": step_dict.get("type"),
            "tags": step_dict.get("tags"),
            "metadata": metadata,
        }
        if step_dict.get("input"):
            step["input"] = {"content": step_dict.get("input")}
        if step_dict.get("output"):
            step["output"] = {"content": step_dict.get("output")}
        if step_dict.get("isError"):
            step["error"] = step_dict.get("output")

        print("\n\n\n")
        print("Step: ", step)
        print("\n\n\n")

        await self.client.api.send_steps([step])
