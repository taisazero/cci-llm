from typing import Any, Dict, List, Optional, Union
import streamlit as st
import argparse
import os
import uuid
import torch
from langchain.llms import OpenAI
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
import requests

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms import HuggingFacePipeline
from langchain.memory import SimpleMemory, ConversationSummaryBufferMemory, CombinedMemory
from langchain.agents import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent, load_tools
from transformers.trainer_utils import set_seed
from transformers import pipeline, StoppingCriteria, StoppingCriteriaList
from langchain.agents import AgentExecutor
from uncc_streaming_agent.base import ConversationalAgent as UNCCAgent

class OpenAIVLLM(ChatOpenAI):
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            
            # "top_p": self.top_p,
            # "frequency_penalty": self.frequency_penalty,
            # "presence_penalty": self.presence_penalty,
            "n": self.n,
            # "request_timeout": self.request_timeout,
            #"logit_bias": self.logit_bias,
        }

        # Azure gpt-35-turbo doesn't support best_of
        # don't specify best_of if it is 1
        # if self.best_of > 1:
        #     normal_params["best_of"] = self.best_of
        if self.max_tokens is not None:
            normal_params["max_tokens"] = self.max_tokens

        return {**normal_params, **self.model_kwargs}

def is_llm_server_active():
    try:
        # Add an api key
        # "aewndfoa1235123"
        response = requests.get("http://localhost:8000/v1/models", headers={"Authorization": "Bearer OnuR-l5IlfYqF8HYoTOYHAcHOXCgL5xASQM5ooGHG6A"})                                                        
        try:
            backup_response = requests.get("http://cci-llm.charlotte.edu/api/v1/models", headers={"Authorization": "Bearer OnuR-l5IlfYqF8HYoTOYHAcHOXCgL5xASQM5ooGHG6A"})                                                        
        except:
            backup_response = requests.Response()
            backup_response.status_code = 404
        if response.status_code == 200:
            print("The VLLM server is running.")
            return response.json()['data'][0]['id'], "http://localhost:8000/v1/models"
        elif backup_response.status_code == 200:
            print("The VLLM server is running.")
            return response.json()['data'][0]['id'], "https://cci-llm.charlotte.edu/api/v1/models"
        else:
            print("The VLLM server is not running.")
            return False, ""
    except requests.exceptions.ConnectionError as e:
        print (e)
        print("Could not connect to the server.")
        return False, ""

class UICallbackHandler(BaseCallbackHandler):
    def __init__(self, prompt_container):
        self.prompt_container = prompt_container
        self.prompt_num = 0

    def on_llm_start(self, _, prompts, **kwargs):
        with self.prompt_container:
            for prompt in prompts:
                self.prompt_num += 1
                st.markdown(f"#### Prompt {self.prompt_num}")
                st.markdown(prompt)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

class SetToolMemoryHandler(BaseCallbackHandler):
    def __init__(self, tool_memory):
        self.tool_memory = tool_memory

    def on_tool_end(self, output, **kwargs):
        if "No results found" not in output:
            self.tool_memory.memories["tool_history"] = f"{kwargs['name']}:\n{output}"

class Streamlit_User_Session:
    def __init__(self, resources, session_id, random_state, session_state=None):
        self.resources = resources
        self.session_id = session_id
        self.random_state = random_state
        self.session_state = session_state
        # dialogue history
        self.messages = [{"role": "assistant", "content": "Hello! How may I help you?"}]

        # setup tool retriever
        # self.tool_retriever = self.resources.tool_vector_store.as_retriever(search_kwargs={"k": 3}) # was 

        # setup persistent memory for tool output
        # tool_memory = SimpleMemory()
        # tool_memory.memories["tool_history"] = ""
        # tool_memory_handler = SetToolMemoryHandler(tool_memory)

    

        # create a tool for each intent
        self.tools = []
        
        # initialize LLM
        # load openai model if llm server is running
        print(f"Initializing with")
        self.llm = OpenAIVLLM(
                model = resources, 
                max_tokens = session_state['max_tokens'],
                temperature = session_state['temperature'],
                #top_p = session_state['top_p'],
                streaming = True, 
            )

        # initialize memory
        conversational_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            input_key='input',
            memory_key='chat_history',
            max_token_limit= 550,
            return_messages= True,
            human_prefix= "User",
            ai_prefix= "Assistant"
        )
       
        if session_state['react']:
            agent = UNCCAgent.from_llm_and_tools(
                self.llm, 
                self.tools, 
                tool_getter=self._get_tools,
                human_prefix="User", 
                ai_prefix="AI", 
            )
            self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent, 
                tools=self.tools,
                verbose=True,
                max_iterations=4,
                early_stopping_method='generate',
                memory=conversational_memory,
                handle_parsing_errors=True,
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )

            chain = (
                RunnablePassthrough.assign(
                    chat_history=RunnableLambda(conversational_memory.load_memory_variables) | itemgetter("chat_history")
                )
            | prompt
            | self.llm
            )
            self.agent_executor = chain

        
    def _get_tools(self, query):
        docs = self.tool_retriever.get_relevant_documents(query)
        return [self.tools[d.metadata["index"]] for d in docs]
    

    def get_response(self, user_input, callbacks=None):
        self.messages.append({"role": "user", "content": user_input})
        if self.random_state is not None:
            set_seed(self.random_state)
        
        if self.session_state['react']:
            response = self.agent_executor.run(user_input, callbacks = callbacks)
        else:
            inputs = {"input": user_input}
            response = self.agent_executor.invoke(input = inputs).content

        # remove \nUser if response ends with it
        response = response.replace("\nUser", "").strip()
        self.messages.append({"role": "assistant", "content": response})
       
        return response
        

st.set_page_config(page_title="UNCC LLM", layout="wide", page_icon="🤖")

@st.cache_resource()
def get_session(_args, session_id, _session_state):
    resources = get_resources(_args)
    return Streamlit_User_Session(resources, session_id, _args.random_state, _session_state)


@st.cache_resource(max_entries=1)
def get_resources(_args):
    os.environ["OPENAI_API_KEY"] = "OnuR-l5IlfYqF8HYoTOYHAcHOXCgL5xASQM5ooGHG6A"
    
    load_local_models, endpoint = is_llm_server_active()

    os.environ["OPENAI_API_BASE"] = endpoint
  
    return load_local_models if load_local_models  else "No Model"
    

def run(args):


    # session info
    if 'react' not in st.session_state:
        st.session_state['react'] = True

    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.0
    if 'max_tokens' not in st.session_state:
        st.session_state['max_tokens'] = 300
    # if 'top_p' not in st.session_state:
    #     st.session_state['top_p'] = 0.7
    
    url_params = st.experimental_get_query_params()
    session_id = url_params.get("session_id", [None])[0]
    session = get_session(args, session_id, st.session_state)
    # sidebar
    with st.sidebar:
        st.image("images/uncc.png")
        st.title("UNCC LLM")
        st.markdown("Welcome to the UNCC LLM interface. You can chat with the assistant here.")
        with st.expander("About"):
            st.markdown(f"This interface is powered by the following large language model: `{session.resources}`. This LLM is hosted by the College of Computing and Informatics at the University of North Carolina at Charlotte.")
            st.markdown("For any questions or feedback, please contact Erfan Al-Hossami at [ealhossa@uncc.edu](mailto:ealhossa@uncc.edu).")
        if st.button("New Session") or session_id is None:
            session_id = str(uuid.uuid4())
            url_params["session_id"] = [session_id]
            st.experimental_set_query_params(**url_params)

        
#        st.session_state['react'] = st.toggle("Enable ReAct Prompting", value=False)
        st.session_state['temperature'] = st.slider("Temperature", value=0.0, min_value=0.0, max_value=1.0, step=0.1)
        st.session_state['max_tokens'] = st.slider("Max Tokens", value=300, min_value=50, max_value=2000, step=100)
        # st.session_state['top_p'] = st.slider("top_p", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
    
    
    # model_tab, data_tab,

    # main body
    user_input = st.chat_input()

    for message in session.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input:
        # user input
        with st.chat_message("user"):
            st.markdown(user_input)

        # intermediate thoughts
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = session.get_response(user_input, callbacks=[st_callback])

        # final message
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the streamlit UI")
    parser.add_argument("--load-in-8bit", action="store_true", default=False)
    parser.add_argument("--random-state", type=int, default=42)

    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)


    run(args)
