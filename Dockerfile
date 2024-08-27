FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./setup.py /code/setup.py

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /code/requirements.txt
RUN pip install -e .

COPY . /code

# List the contents of the /code directory to verify files are copied correctly
RUN ls -R /code

# Change permissions to allow writing to the directory
RUN chmod -R 777 /code

# Create a logs directory and set permissions
RUN mkdir /code/apps/ai_tutor/logs && chmod 777 /code/apps/ai_tutor/logs

# Create a cache directory within the application's working directory
RUN mkdir /.cache && chmod -R 777 /.cache

WORKDIR /code/apps/ai_tutor

# Expose the port the app runs on
EXPOSE 7860

RUN --mount=type=secret,id=HUGGINGFACEHUB_API_TOKEN,mode=0444,required=true 
RUN --mount=type=secret,id=OPENAI_API_KEY,mode=0444,required=true 
RUN --mount=type=secret,id=CHAINLIT_URL,mode=0444,required=true 
RUN --mount=type=secret,id=LITERAL_API_URL,mode=0444,required=true 
RUN --mount=type=secret,id=LLAMA_CLOUD_API_KEY,mode=0444,required=true 
RUN --mount=type=secret,id=OAUTH_GOOGLE_CLIENT_ID,mode=0444,required=true 
RUN --mount=type=secret,id=OAUTH_GOOGLE_CLIENT_SECRET,mode=0444,required=true 
RUN --mount=type=secret,id=LITERAL_API_KEY_LOGGING,mode=0444,required=true 
RUN --mount=type=secret,id=CHAINLIT_AUTH_SECRET,mode=0444,required=true 

# Default command to run the application
CMD python -m edubotics_core.vectorstore.store_manager --config_file config/config.yml --project_config_file config/project_config.yml && python -m uvicorn app:app --host 0.0.0.0 --port 7860