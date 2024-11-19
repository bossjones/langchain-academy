import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Chatbot with Profile Schema 

        ## Review

        We introduced the [LangGraph Memory Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) as a way to save and retrieve long-term memories.

        We built a simple chatbot that uses both `short-term (within-thread)` and `long-term (across-thread)` memory.

        It saved long-term [semantic memory](https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory) (facts about the user) ["in the hot path"](https://langchain-ai.github.io/langgraph/concepts/memory/#writing-memories), as the user is chatting with it.

        ## Goals

        Our chatbot saved memories as a string. In practice, we often want memories to have a structure. 
         
        For example, memories can be a [single, continuously updated schema]((https://langchain-ai.github.io/langgraph/concepts/memory/#profile)). 
         
        In our case, we want this to be a single user profile.
         
        We'll extend our chatbot to save semantic memories to a single [user profile](https://langchain-ai.github.io/langgraph/concepts/memory/#profile). 

        We'll also introduce a library, [Trustcall](https://github.com/hinthornw/trustcall), to update this schema with new information. 
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
        ## Defining a user profile schema

        Python has many different types for [structured data](https://python.langchain.com/docs/concepts/structured_outputs/#schema-definition), such as TypedDict, Dictionaries, JSON, and [Pydantic](https://docs.pydantic.dev/latest/). 

        Let's start by using TypedDict to define a user profile schema.
        """
    )
    return


@app.cell
def __():
    from typing import TypedDict, List

    class UserProfile(TypedDict):
        """User profile schema with typed fields"""
        user_name: str  # The user's preferred name
        interests: List[str]  # A list of the user's interests
    return List, TypedDict, UserProfile


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Saving a schema to the store

        The [LangGraph Store](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) accepts any Python dictionary as the `value`. 
        """
    )
    return


@app.cell
def __(UserProfile):
    # TypedDict instance
    user_profile: UserProfile = {
        "user_name": "Lance",
        "interests": ["biking", "technology", "coffee"]
    }
    user_profile
    return (user_profile,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We use the [put](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.put) method to save the TypedDict to the store.
        """
    )
    return


@app.cell
def __(user_id, user_profile):
    import uuid
    from langgraph.store.memory import InMemoryStore
    in_memory_store = InMemoryStore()
    _user_id = '1'
    namespace_for_memory = (user_id, 'memory')
    key = 'user_profile'
    value = user_profile
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
        We use [search](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search) to retrieve objects from the store by namespace.
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
        We can also use [get](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get) to retrieve a specific object by namespace and key.
        """
    )
    return


@app.cell
def __(in_memory_store, namespace_for_memory):
    # Get the memory by namespace and key
    profile = in_memory_store.get(namespace_for_memory, "user_profile")
    profile.value
    return (profile,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Chatbot with profile schema

        Now we know how to specify a schema for the memories and save it to the store.

        Now, how do we actually *create* memories with this particular schema?

        In our chatbot, we [want to create memories from a user chat](https://langchain-ai.github.io/langgraph/concepts/memory/#profile). 

        This is where the concept of [structured outputs](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage) is useful. 

        LangChain's [chat model](https://python.langchain.com/docs/concepts/chat_models/) interface has a [`with_structured_output`](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage) method to enforce structured output.

        This is useful when we want to enforce that the output conforms to a schema, and it parses the output for us.
        """
    )
    return


@app.cell
def __():
    _set_env("OPENAI_API_KEY")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's pass the `UserProfile` schema we created to the `with_structured_output` method.

        We can then invoke the chat model with a list of [messages](https://python.langchain.com/docs/concepts/messages/) and get a structured output that conforms to our schema.
        """
    )
    return


@app.cell
def __(UserProfile):
    from pydantic import BaseModel, Field

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Bind schema to model
    model_with_structure = model.with_structured_output(UserProfile)

    # Invoke the model to produce structured output that matches the schema
    structured_output = model_with_structure.invoke([HumanMessage("My name is Lance, I like to bike.")])
    structured_output
    return (
        BaseModel,
        ChatOpenAI,
        Field,
        HumanMessage,
        model,
        model_with_structure,
        structured_output,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's use this with our chatbot.

        This only requires minor changes to the `write_memory` function. 

        We use `model_with_structure`, as defined above, to produce a profile that matches our schema. 
        """
    )
    return


@app.cell
def __(
    InMemoryStore,
    MODEL_SYSTEM_MESSAGE,
    builder,
    model,
    model_with_structure,
    within_thread_memory,
):
    from IPython.display import Image, display
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.store.base import BaseStore
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.runnables.config import RunnableConfig
    _MODEL_SYSTEM_MESSAGE = 'You are a helpful assistant with memory that provides information about the user.\nIf you have memory for this user, use it to personalize your responses.\nHere is the memory (it may be empty): {memory}'
    CREATE_MEMORY_INSTRUCTION = "Create or update a user profile memory based on the user's chat history.\nThis will be saved for long-term memory. If there is an existing memory, simply update it.\nHere is the existing memory (it may be empty): {memory}"

    def _call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Load memory from the store and use it to personalize the chatbot's response."""
        user_id = config['configurable']['user_id']
        namespace = ('memory', user_id)
        existing_memory = store.get(namespace, 'user_memory')
        if existing_memory and existing_memory.value:
            memory_dict = existing_memory.value
            formatted_memory = f"Name: {memory_dict.get('user_name', 'Unknown')}\nInterests: {', '.join(memory_dict.get('interests', []))}"
        else:
            formatted_memory = None
        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)
        response = model.invoke([SystemMessage(content=system_msg)] + state['messages'])
        return {'messages': response}

    def _write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and save a memory to the store."""
        user_id = config['configurable']['user_id']
        namespace = ('memory', user_id)
        existing_memory = store.get(namespace, 'user_memory')
        if existing_memory and existing_memory.value:
            memory_dict = existing_memory.value
            formatted_memory = f"Name: {memory_dict.get('user_name', 'Unknown')}\nInterests: {', '.join(memory_dict.get('interests', []))}"
        else:
            formatted_memory = None
        system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=formatted_memory)
        new_memory = model_with_structure.invoke([SystemMessage(content=system_msg)] + state['messages'])
        key = 'user_memory'
        store.put(namespace, key, new_memory)
    _builder = StateGraph(MessagesState)
    _builder.add_node('call_model', _call_model)
    _builder.add_node('write_memory', _write_memory)
    _builder.add_edge(START, 'call_model')
    _builder.add_edge('call_model', 'write_memory')
    _builder.add_edge('write_memory', END)
    across_thread_memory = InMemoryStore()
    _within_thread_memory = MemorySaver()
    graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)
    display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    return (
        AIMessage,
        BaseStore,
        CREATE_MEMORY_INSTRUCTION,
        END,
        HumanMessage,
        Image,
        MemorySaver,
        MessagesState,
        RunnableConfig,
        START,
        StateGraph,
        SystemMessage,
        across_thread_memory,
        display,
        graph,
    )


@app.cell
def __(HumanMessage_1, graph):
    config = {'configurable': {'thread_id': '1', 'user_id': '1'}}
    _input_messages = [HumanMessage_1(content='Hi, my name is Lance and I like to bike around San Francisco and eat at bakeries.')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return (config,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's check the memory in the store. 

        We can see that the memory is a dictionary that matches our schema.
        """
    )
    return


@app.cell
def __(across_thread_memory, namespace, user_id):
    _user_id = '1'
    _namespace = ('memory', user_id)
    existing_memory = across_thread_memory.get(namespace, 'user_memory')
    existing_memory.value
    return (existing_memory,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## When can this fail?

        [`with_structured_output`](https://python.langchain.com/docs/concepts/structured_outputs/#recommended-usage) is very useful, but what happens if we're working with a more complex schema? 

        [Here's](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) an example of a more complex schema, which we'll test below. 

        This is a [Pydantic](https://docs.pydantic.dev/latest/) model that describes a user's preferences for communication and trust fall.
        """
    )
    return


@app.cell
def __(BaseModel, List_1):
    from typing import List, Optional

    class OutputFormat(BaseModel):
        preference: str
        sentence_preference_revealed: str

    class TelegramPreferences(BaseModel):
        preferred_encoding: Optional[List_1[OutputFormat]] = None
        favorite_telegram_operators: Optional[List_1[OutputFormat]] = None
        preferred_telegram_paper: Optional[List_1[OutputFormat]] = None

    class MorseCode(BaseModel):
        preferred_key_type: Optional[List_1[OutputFormat]] = None
        favorite_morse_abbreviations: Optional[List_1[OutputFormat]] = None

    class Semaphore(BaseModel):
        preferred_flag_color: Optional[List_1[OutputFormat]] = None
        semaphore_skill_level: Optional[List_1[OutputFormat]] = None

    class TrustFallPreferences(BaseModel):
        preferred_fall_height: Optional[List_1[OutputFormat]] = None
        trust_level: Optional[List_1[OutputFormat]] = None
        preferred_catching_technique: Optional[List_1[OutputFormat]] = None

    class CommunicationPreferences(BaseModel):
        telegram: TelegramPreferences
        morse_code: MorseCode
        semaphore: Semaphore

    class UserPreferences(BaseModel):
        communication_preferences: CommunicationPreferences
        trust_fall_preferences: TrustFallPreferences

    class TelegramAndTrustFallPreferences(BaseModel):
        pertinent_user_preferences: UserPreferences
    return (
        CommunicationPreferences,
        List,
        MorseCode,
        Optional,
        OutputFormat,
        Semaphore,
        TelegramAndTrustFallPreferences,
        TelegramPreferences,
        TrustFallPreferences,
        UserPreferences,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's try extraction of this schema using the `with_structured_output` method.
        """
    )
    return


@app.cell
def __(TelegramAndTrustFallPreferences, model):
    from pydantic import ValidationError
    model_with_structure_1 = model.with_structured_output(TelegramAndTrustFallPreferences)
    conversation = 'Operator: How may I assist with your telegram, sir?\nCustomer: I need to send a message about our trust fall exercise.\nOperator: Certainly. Morse code or standard encoding?\nCustomer: Morse, please. I love using a straight key.\nOperator: Excellent. What\'s your message?\nCustomer: Tell him I\'m ready for a higher fall, and I prefer the diamond formation for catching.\nOperator: Done. Shall I use our "Daredevil" paper for this daring message?\nCustomer: Perfect! Send it by your fastest carrier pigeon.\nOperator: It\'ll be there within the hour, sir.'
    try:
        model_with_structure_1.invoke(f'Extract the preferences from the following conversation:\n    <convo>\n    {conversation}\n    </convo>')
    except ValidationError as e:
        print(e)
    return ValidationError, conversation, model_with_structure_1


@app.cell
def __(mo):
    mo.md(
        r"""
        If we naively extract more complex schemas, even using high capacity model like `gpt-4o`, it is prone to failure.

        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Trustcall for creating and updating profile schemas

        As we can see, working with schemas can be tricky.

        Complex schemas can be difficult to extract. 

        In addition, updating even simple schemas can pose challenges.

        Consider our above chatbot. 

        We regenerated the profile schema *from scratch* each time we chose to save a new memory.

        This is inefficient, potentially wasting model tokens if the schema contains a lot of information to re-generate each time.

        Worse, we may loose information when regenerating the profile from scratch.

        Addressing these problems is the motivation for [TrustCall](https://github.com/hinthornw/trustcall)!

        This is an open-source library for updating JSON schemas developed by one [Will Fu-Hinthorn](https://github.com/hinthornw) on the LangChain team.

        It's motivated by exactly these challenges while working on memory.

        Let's first show simple usage of extraction with TrustCall on this list of [messages](https://python.langchain.com/docs/concepts/messages/).

        """
    )
    return


@app.cell
def __(AIMessage, HumanMessage_1):
    conversation_1 = [HumanMessage_1(content="Hi, I'm Lance."), AIMessage(content='Nice to meet you, Lance.'), HumanMessage_1(content='I really like biking around San Francisco.')]
    return (conversation_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We use `create_extractor`, passing in the model as well as our schema as a [tool](https://python.langchain.com/docs/concepts/tools/).

        With TrustCall, can supply supply the schema in various ways. 

        For example, we can pass a JSON object / Python dictionary or Pydantic model.

        Under the hood, TrustCall uses [tool calling](https://python.langchain.com/docs/concepts/tool_calling/) to produce [structured output](https://python.langchain.com/docs/concepts/structured_outputs/) from an input list of [messages](https://python.langchain.com/docs/concepts/messages/).

        To force Trustcall to produce [structured output](https://python.langchain.com/docs/concepts/structured_outputs/), we can include the schema name in the `tool_choice` argument.

        We can invoke the extractor with  the above conversation.
        """
    )
    return


@app.cell
def __(
    BaseModel,
    ChatOpenAI,
    Field,
    List_1,
    SystemMessage,
    conversation_1,
    system_msg,
):
    from trustcall import create_extractor

    class UserProfile_1(BaseModel):
        """User profile schema with typed fields"""
        user_name: str = Field(description="The user's preferred name")
        interests: List_1[str] = Field(description="A list of the user's interests")
    model_1 = ChatOpenAI(model='gpt-4o', temperature=0)
    trustcall_extractor = create_extractor(model_1, tools=[UserProfile_1], tool_choice='UserProfile')
    _system_msg = 'Extract the user profile from the following conversation'
    result = trustcall_extractor.invoke({'messages': [SystemMessage(content=system_msg)] + conversation_1})
    return (
        UserProfile_1,
        create_extractor,
        model_1,
        result,
        trustcall_extractor,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        When we invoke the extractor, we get a few things:

        * `messages`: The list of `AIMessages` that contain the tool calls. 
        * `responses`: The resulting parsed tool calls that match our schema.
        * `response_metadata`: Applicable if updating existing tool calls. It says which of the responses correspond to which of the existing objects.

        """
    )
    return


@app.cell
def __(result):
    for _m in result['messages']:
        _m.pretty_print()
    return


@app.cell
def __(result):
    schema = result["responses"]
    schema
    return (schema,)


@app.cell
def __(schema):
    schema[0].model_dump()
    return


@app.cell
def __(result):
    result["response_metadata"]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's see how we can use it to *update* the profile.

        For updating, TrustCall takes a set of messages as well as the existing schema. 

        The central idea is that it prompts the model to produce a [JSON Patch](https://jsonpatch.com/) to update only the relevant parts of the schema.

        This is less error-prone than naively overwriting the entire schema.

        It's also more efficient since the model only needs to generate the parts of the schema that have changed.

        We can save the existing schema as a dict.

        We can use `model_dump()` to serialize a Pydantic model instance into a dict. 

        We pass it to the `"existing"` argument along with the schema name, `UserProfile`. 
        """
    )
    return


@app.cell
def __(
    AIMessage,
    HumanMessage_1,
    SystemMessage,
    schema,
    system_msg,
    trustcall_extractor,
):
    updated_conversation = [HumanMessage_1(content="Hi, I'm Lance."), AIMessage(content='Nice to meet you, Lance.'), HumanMessage_1(content='I really like biking around San Francisco.'), AIMessage(content='San Francisco is a great city! Where do you go after biking?'), HumanMessage_1(content='I really like to go to a bakery after biking.')]
    _system_msg = f'Update the memory (JSON doc) to incorporate new information from the following conversation'
    result_1 = trustcall_extractor.invoke({'messages': [SystemMessage(content=system_msg)] + updated_conversation}, {'existing': {'UserProfile': schema[0].model_dump()}})
    return result_1, updated_conversation


@app.cell
def __(result_1):
    for _m in result_1['messages']:
        _m.pretty_print()
    return


@app.cell
def __(result_1):
    result_1['response_metadata']
    return


@app.cell
def __(result_1):
    updated_schema = result_1['responses'][0]
    updated_schema.model_dump()
    return (updated_schema,)


@app.cell
def __(mo):
    mo.md(
        r"""
        LangSmith trace:

        https://smith.langchain.com/public/229eae22-1edb-44c6-93e6-489124a43968/r

        Now, let's also test Trustcall on the [challenging schema](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) that we saw earlier.
        """
    )
    return


@app.cell
def __(TelegramAndTrustFallPreferences, create_extractor, model_1):
    bound = create_extractor(model_1, tools=[TelegramAndTrustFallPreferences], tool_choice='TelegramAndTrustFallPreferences')
    conversation_2 = 'Operator: How may I assist with your telegram, sir?\nCustomer: I need to send a message about our trust fall exercise.\nOperator: Certainly. Morse code or standard encoding?\nCustomer: Morse, please. I love using a straight key.\nOperator: Excellent. What\'s your message?\nCustomer: Tell him I\'m ready for a higher fall, and I prefer the diamond formation for catching.\nOperator: Done. Shall I use our "Daredevil" paper for this daring message?\nCustomer: Perfect! Send it by your fastest carrier pigeon.\nOperator: It\'ll be there within the hour, sir.'
    result_2 = bound.invoke(f'Extract the preferences from the following conversation:\n<convo>\n{conversation_2}\n</convo>')
    result_2['responses'][0]
    return bound, conversation_2, result_2


@app.cell
def __(mo):
    mo.md(
        r"""
        Trace: 

        https://smith.langchain.com/public/5cd23009-3e05-4b00-99f0-c66ee3edd06e/r

        For more examples, you can see an overview video [here](https://www.youtube.com/watch?v=-H4s0jQi-QY).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Chatbot with profile schema updating

        Now, let's bring Trustcall into our chatbot to create *and update* a memory profile.
        """
    )
    return


@app.cell
def __(
    BaseModel,
    BaseStore,
    ChatOpenAI,
    END,
    Field,
    Image,
    InMemoryStore,
    MODEL_SYSTEM_MESSAGE,
    MemorySaver,
    MessagesState,
    RunnableConfig,
    START,
    StateGraph,
    SystemMessage_1,
    builder,
    create_extractor,
    display,
    within_thread_memory,
):
    from langchain_core.messages import HumanMessage, SystemMessage
    model_2 = ChatOpenAI(model='gpt-4o', temperature=0)

    class UserProfile_2(BaseModel):
        """ Profile of a user """
        user_name: str = Field(description="The user's preferred name")
        user_location: str = Field(description="The user's location")
        interests: list = Field(description="A list of the user's interests")
    trustcall_extractor_1 = create_extractor(model_2, tools=[UserProfile_2], tool_choice='UserProfile')
    _MODEL_SYSTEM_MESSAGE = 'You are a helpful assistant with memory that provides information about the user.\nIf you have memory for this user, use it to personalize your responses.\nHere is the memory (it may be empty): {memory}'
    TRUSTCALL_INSTRUCTION = 'Create or update the memory (JSON doc) to incorporate information from the following conversation:'

    def _call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Load memory from the store and use it to personalize the chatbot's response."""
        user_id = config['configurable']['user_id']
        namespace = ('memory', user_id)
        existing_memory = store.get(namespace, 'user_memory')
        if existing_memory and existing_memory.value:
            memory_dict = existing_memory.value
            formatted_memory = f"Name: {memory_dict.get('user_name', 'Unknown')}\nLocation: {memory_dict.get('user_location', 'Unknown')}\nInterests: {', '.join(memory_dict.get('interests', []))}"
        else:
            formatted_memory = None
        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)
        response = model_2.invoke([SystemMessage_1(content=system_msg)] + state['messages'])
        return {'messages': response}

    def _write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and save a memory to the store."""
        user_id = config['configurable']['user_id']
        namespace = ('memory', user_id)
        existing_memory = store.get(namespace, 'user_memory')
        existing_profile = {'UserProfile': existing_memory.value} if existing_memory else None
        result = trustcall_extractor_1.invoke({'messages': [SystemMessage_1(content=TRUSTCALL_INSTRUCTION)] + state['messages'], 'existing': existing_profile})
        updated_profile = result['responses'][0].model_dump()
        key = 'user_memory'
        store.put(namespace, key, updated_profile)
    _builder = StateGraph(MessagesState)
    _builder.add_node('call_model', _call_model)
    _builder.add_node('write_memory', _write_memory)
    _builder.add_edge(START, 'call_model')
    _builder.add_edge('call_model', 'write_memory')
    _builder.add_edge('write_memory', END)
    across_thread_memory_1 = InMemoryStore()
    _within_thread_memory = MemorySaver()
    graph_1 = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory_1)
    display(Image(graph_1.get_graph(xray=1).draw_mermaid_png()))
    return (
        HumanMessage,
        SystemMessage,
        TRUSTCALL_INSTRUCTION,
        UserProfile_2,
        across_thread_memory_1,
        graph_1,
        model_2,
        trustcall_extractor_1,
    )


@app.cell
def __(HumanMessage_2, graph_1):
    config_1 = {'configurable': {'thread_id': '1', 'user_id': '1'}}
    _input_messages = [HumanMessage_2(content='Hi, my name is Lance')]
    for _chunk in graph_1.stream({'messages': _input_messages}, config_1, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return (config_1,)


@app.cell
def __(HumanMessage_2, config_1, graph_1):
    _input_messages = [HumanMessage_2(content='I like to bike around San Francisco')]
    for _chunk in graph_1.stream({'messages': _input_messages}, config_1, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(across_thread_memory_1, namespace, user_id):
    _user_id = '1'
    _namespace = ('memory', user_id)
    existing_memory_1 = across_thread_memory_1.get(namespace, 'user_memory')
    existing_memory_1.dict()
    return (existing_memory_1,)


@app.cell
def __(existing_memory_1):
    existing_memory_1.value
    return


@app.cell
def __(HumanMessage_2, config_1, graph_1):
    _input_messages = [HumanMessage_2(content='I also enjoy going to bakeries')]
    for _chunk in graph_1.stream({'messages': _input_messages}, config_1, stream_mode='values'):
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
def __(HumanMessage_2, graph_1):
    config_2 = {'configurable': {'thread_id': '2', 'user_id': '1'}}
    _input_messages = [HumanMessage_2(content='What bakeries do you recommend for me?')]
    for _chunk in graph_1.stream({'messages': _input_messages}, config_2, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return (config_2,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Trace:

        https://smith.langchain.com/public/f45bdaf0-6963-4c19-8ec9-f4b7fe0f68ad/r

        ## Studio

        ![Screenshot 2024-10-30 at 11.26.31 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6732d0437060f1754ea79908_Screenshot%202024-11-11%20at%207.48.53%E2%80%AFPM.png)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

