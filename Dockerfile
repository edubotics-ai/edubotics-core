FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install --no-cache-dir transformers==4.36.2 torch==2.1.2

RUN pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.2.32

COPY . /code

RUN ls -R

# Change permissions to allow writing to the directory
RUN chmod -R 777 /code

# Create a logs directory and set permissions
RUN mkdir /code/logs && chmod 777 /code/logs

# Create a cache directory within the application's working directory
RUN mkdir /.cache && chmod -R 777 /.cache

RUN --mount=type=secret,id=HUGGINGFACEHUB_API_TOKEN,mode=0444,required=true 
RUN --mount=type=secret,id=OPENAI_API_KEY,mode=0444,required=true 

CMD python code/modules/vector_db.py && chainlit run code/main.py --host 0.0.0.0 --port 7860