import os
from literalai import LiteralClient
from openai import OpenAI

# Use these for manual testing
# os.environ['OPENAI_API_KEY'] = ""
# os.environ['LITERAL_API_KEY'] = ""

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
literal_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))
literal_client.instrument_openai()

@literal_client.step(type="run")
def ai_assistant(user_query: str):
    completion = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": user_query,
                    }
                ],
            )
    literal_client.message(content=completion.choices[0].message.content, type="assistant_message", name="AI Tutor")

def main():


    with literal_client.thread(name="DS 598 B1") as thread: # pass name="Example Name" to give thread a name
        # when you receive user, you get createdAt, id, identifier, metadata.
        # you have to use id not the identifier as id is the UUID
        user = literal_client.api.get_user(identifier="000001") # hardcoded uid
        if user == None: # this should be removed if setting identifier manually
            user = literal_client.api.create_user(identifier="000001")
        
        thread.participant_id = user.id
        thread.tags = ["gpt-3.5-turbo", "freshman", "v-0.0.606"] # add tags here
        thread.metadata = {"year": "freshman"} # corrected metadata to be a dictionary


        student_question = "Can you do 1+1 for me?"
        literal_client.message(content=student_question, type="user_message", name="Student", metadata={"pdf":"2"}) # shows individual metadata
        ai_assistant(student_question)
        
        # The second question is not persistant.
        # student_question_2 = "Hello!"
        # literal_client.message(content=student_question_2, type="user_message", name="Student", metadata={"pdf":"3"}) # shows individual metadata
        # ai_assistant(student_question_2)

main()

literal_client.flush_and_stop() # end and submit to literalai
