import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Chatbot with Collection Schema 

        ## Review

        We extended our chatbot to save semantic memories to a single [user profile](https://langchain-ai.github.io/langgraph/concepts/memory/#profile). 

        We also introduced a library, [Trustcall](https://github.com/hinthornw/trustcall), to update this schema with new information. 

        ## Goals

        Sometimes we want to save memories to a [collection](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_200) rather than single profile. 

        Here we'll update our chatbot to [save memories to a collection](https://langchain-ai.github.io/langgraph/concepts/memory/#collection).

        We'll also show how to use [Trustcall](https://github.com/hinthornw/trustcall) to update this collection. 

        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # # %pip install -U langchain_openai langgraph trustcall langchain_core
    return


@app.cell
def __():
    import os, getpass

    def _set_env(var: str):
        # Check if the variable is set in the OS environment
        env_value = os.environ.get(var)
        if not env_value:
            # If not set, prompt the user for input
            env_value = getpass.getpass(f"{var}: ")

        # Set the environment variable for the current process
        os.environ[var] = env_value

    _set_env("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"
    return getpass, os


@app.cell
def __(os_1):
    from __future__ import annotations
    import os
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='.env', override=True)
    os_1.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os_1.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    return annotations, load_dotenv, os


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Defining a collection schema

        Instead of storing user information in a fixed profile structure, we'll create a flexible collection schema to store memories about user interactions.

        Each memory will be stored as a separate entry with a single `content` field for the main information we want to remember

        This approach allows us to build an open-ended collection of memories that can grow and change as we learn more about the user.

        We can define a collection schema as a [Pydantic](https://docs.pydantic.dev/latest/) object. 
        """
    )
    return


@app.cell
def __():
    from pydantic import BaseModel, Field

    class Memory(BaseModel):
        content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

    class MemoryCollection(BaseModel):
        memories: list[Memory] = Field(description="A list of memories about the user.")
    return BaseModel, Field, Memory, MemoryCollection


@app.cell
def __():
    _set_env("OPENAI_API_KEY")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can used LangChain's chat model [chat model](https://python.langchain.com/docs/concepts/chat_models/) interface's [`with_structured_output`](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage) method to enforce structured output.
        """
    )
    return


@app.cell
def __(MemoryCollection):
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Bind schema to model
    model_with_structure = model.with_structured_output(MemoryCollection)

    # Invoke the model to produce structured output that matches the schema
    memory_collection = model_with_structure.invoke([HumanMessage("My name is Lance. I like to bike.")])
    memory_collection.memories
    return (
        ChatOpenAI,
        HumanMessage,
        memory_collection,
        model,
        model_with_structure,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        We can use `model_dump()` to serialize a Pydantic model instance into a Python dictionary.
        """
    )
    return


@app.cell
def __(memory_collection):
    memory_collection.memories[0].model_dump()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Save dictionary representation of each memory to the store. 
        """
    )
    return


@app.cell
def __(memory_collection, user_id):
    import uuid
    from langgraph.store.memory import InMemoryStore
    in_memory_store = InMemoryStore()
    _user_id = '1'
    namespace_for_memory = (user_id, 'memories')
    key = str(uuid.uuid4())
    value = memory_collection.memories[0].model_dump()
    in_memory_store.put(namespace_for_memory, key, value)
    key = str(uuid.uuid4())
    value = memory_collection.memories[1].model_dump()
    in_memory_store.put(namespace_for_memory, key, value)
    return (
        InMemoryStore,
        in_memory_store,
        key,
        namespace_for_memory,
        uuid,
        value,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Search for memories in the store. 
        """
    )
    return


@app.cell
def __(in_memory_store, namespace_for_memory):
    for _m in in_memory_store.search(namespace_for_memory):
        print(_m.dict())
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Updating collection schema

        We discussed the challenges with updating a profile schema in the last lesson. 

        The same applies for collections! 

        We want the ability to update the collection with new memories as well as update existing memories in the collection. 

        Now we'll show that [Trustcall](https://github.com/hinthornw/trustcall) can be also used to update a collection. 

        This enables both addition of new memories as well as [updating existing memories in the collection](https://github.com/hinthornw/trustcall?tab=readme-ov-file#simultanous-updates--insertions
        ).

        Let's define a new extractor with Trustcall. 

        As before, we provide the schema for each memory, `Memory`.  

        But, we can supply `enable_inserts=True` to allow the extractor to insert new memories to the collection. 
        """
    )
    return


@app.cell
def __(Memory, model):
    from trustcall import create_extractor

    # Create the extractor
    trustcall_extractor = create_extractor(
        model,
        tools=[Memory],
        tool_choice="Memory",
        enable_inserts=True,
    )
    return create_extractor, trustcall_extractor


@app.cell
def __(HumanMessage_1, trustcall_extractor):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    instruction = 'Extract memories from the following conversation:'
    conversation = [HumanMessage_1(content="Hi, I'm Lance."), AIMessage(content='Nice to meet you, Lance.'), HumanMessage_1(content='This morning I had a nice bike ride in San Francisco.')]
    result = trustcall_extractor.invoke({'messages': [SystemMessage(content=instruction)] + conversation})
    return (
        AIMessage,
        HumanMessage,
        SystemMessage,
        conversation,
        instruction,
        result,
    )


@app.cell
def __(result):
    for _m in result['messages']:
        _m.pretty_print()
    return


@app.cell
def __(result):
    for _m in result['responses']:
        print(_m)
    return


@app.cell
def __(result):
    for _m in result['response_metadata']:
        print(_m)
    return


@app.cell
def __(AIMessage, HumanMessage_1, result):
    updated_conversation = [AIMessage(content="That's great, did you do after?"), HumanMessage_1(content='I went to Tartine and ate a croissant.'), AIMessage(content='What else is on your mind?'), HumanMessage_1(content='I was thinking about my Japan, and going back this winter!')]
    system_msg = 'Update existing memories and create new ones based on the following conversation:'
    tool_name = 'Memory'
    existing_memories = [(str(i), tool_name, memory.model_dump()) for i, memory in enumerate(result['responses'])] if result['responses'] else None
    existing_memories
    return existing_memories, system_msg, tool_name, updated_conversation


@app.cell
def __(existing_memories, trustcall_extractor, updated_conversation):
    result_1 = trustcall_extractor.invoke({'messages': updated_conversation, 'existing': existing_memories})
    return (result_1,)


@app.cell
def __(result_1):
    for _m in result_1['messages']:
        _m.pretty_print()
    return


@app.cell
def __(result_1):
    for _m in result_1['responses']:
        print(_m)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        This tells us that we updated the first memory in the collection by specifying the `json_doc_id`. 
        """
    )
    return


@app.cell
def __(result_1):
    for _m in result_1['response_metadata']:
        print(_m)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        LangSmith trace: 

        https://smith.langchain.com/public/ebc1cb01-f021-4794-80c0-c75d6ea90446/r
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Chatbot with collection schema updating

        Now, let's bring Trustcall into our chatbot to create and update a memory collection.
        """
    )
    return


@app.cell
def __(
    BaseModel,
    ChatOpenAI,
    Field,
    InMemoryStore,
    SystemMessage_1,
    create_extractor,
    uuid,
):
    from IPython.display import Image, display
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import merge_message_runs
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables.config import RunnableConfig
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.base import BaseStore
    model_1 = ChatOpenAI(model='gpt-4o', temperature=0)

    class Memory_1(BaseModel):
        content: str = Field(description='The main content of the memory. For example: User expressed interest in learning about French.')
    trustcall_extractor_1 = create_extractor(model_1, tools=[Memory_1], tool_choice='Memory', enable_inserts=True)
    MODEL_SYSTEM_MESSAGE = 'You are a helpful chatbot. You are designed to be a companion to a user.\n\nYou have a long term memory which keeps track of information you learn about the user over time.\n\nCurrent Memory (may include updated memories from this conversation):\n\n{memory}'
    TRUSTCALL_INSTRUCTION = 'Reflect on following interaction.\n\nUse the provided tools to retain any necessary memories about the user.\n\nUse parallel tool calling to handle updates and insertions simultaneously:'

    def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Load memories from the store and use them to personalize the chatbot's response."""
        user_id = config['configurable']['user_id']
        namespace = ('memories', user_id)
        memories = store.search(namespace)
        info = '\n'.join((f"- {mem.value['content']}" for mem in memories))
        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)
        response = model_1.invoke([SystemMessage_1(content=system_msg)] + state['messages'])
        return {'messages': response}

    def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection."""
        user_id = config['configurable']['user_id']
        namespace = ('memories', user_id)
        existing_items = store.search(namespace)
        tool_name = 'Memory'
        existing_memories = [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items] if existing_items else None
        updated_messages = list(merge_message_runs(messages=[SystemMessage_1(content=TRUSTCALL_INSTRUCTION)] + state['messages']))
        result = trustcall_extractor_1.invoke({'messages': updated_messages, 'existing': existing_memories})
        for r, rmeta in zip(result['responses'], result['response_metadata']):
            store.put(namespace, rmeta.get('json_doc_id', str(uuid.uuid4())), r.model_dump(mode='json'))
    builder = StateGraph(MessagesState)
    builder.add_node('call_model', call_model)
    builder.add_node('write_memory', write_memory)
    builder.add_edge(START, 'call_model')
    builder.add_edge('call_model', 'write_memory')
    builder.add_edge('write_memory', END)
    across_thread_memory = InMemoryStore()
    within_thread_memory = MemorySaver()
    graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)
    display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    return (
        BaseStore,
        END,
        HumanMessage,
        Image,
        MODEL_SYSTEM_MESSAGE,
        MemorySaver,
        Memory_1,
        MessagesState,
        RunnableConfig,
        START,
        StateGraph,
        SystemMessage,
        TRUSTCALL_INSTRUCTION,
        across_thread_memory,
        builder,
        call_model,
        display,
        graph,
        merge_message_runs,
        model_1,
        trustcall_extractor_1,
        within_thread_memory,
        write_memory,
    )


@app.cell
def __(HumanMessage_2, graph):
    config = {'configurable': {'thread_id': '1', 'user_id': '1'}}
    _input_messages = [HumanMessage_2(content='Hi, my name is Lance')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return (config,)


@app.cell
def __(HumanMessage_2, config, graph):
    _input_messages = [HumanMessage_2(content='I like to bike around San Francisco')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(across_thread_memory, user_id):
    _user_id = '1'
    namespace = ('memories', user_id)
    memories = across_thread_memory.search(namespace)
    for _m in memories:
        print(_m.dict())
    return memories, namespace


@app.cell
def __(HumanMessage_2, config, graph):
    _input_messages = [HumanMessage_2(content='I also enjoy going to bakeries')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Continue the conversation in a new thread.
        """
    )
    return


@app.cell
def __(HumanMessage_2, graph):
    config_1 = {'configurable': {'thread_id': '2', 'user_id': '1'}}
    _input_messages = [HumanMessage_2(content='What bakeries do you recommend for me?')]
    for _chunk in graph.stream({'messages': _input_messages}, config_1, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return (config_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### LangSmith 

        https://smith.langchain.com/public/c87543ec-b426-4a82-a3ab-94d01c01d9f4/r

        ## Studio

        ![Screenshot 2024-10-30 at 11.29.25 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6732d0876d3daa19fef993ba_Screenshot%202024-11-11%20at%207.50.21%E2%80%AFPM.png)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

