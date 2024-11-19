import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Memory Agent

        ## Review

        We created a chatbot that saves semantic memories to a single [user profile](https://langchain-ai.github.io/langgraph/concepts/memory/#profile) or [collection](https://langchain-ai.github.io/langgraph/concepts/memory/#collection).

        We introduced [Trustcall](https://github.com/hinthornw/trustcall) as a way to update either schema.

        ## Goals

        Now, we're going to pull together the pieces we've learned to build an [agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/) with long-term memory.

        Our agent, `task_mAIstro`, will help us manage a ToDo list! 

        The chatbots we built previously *always* reflected on the conversation and saved memories. 

        `task_mAIstro` will decide *when* to save memories (items to our ToDo list).

        The chatbots we built previously always saved one type of memory, a profile or a collection. 

        `task_mAIstro` can decide to save to either a user profile or a collection of ToDo items.

        In addition semantic memory, `task_mAIstro` also will manage procedural memory.

        This allows the user to update their preferences for creating ToDo items. 
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
def __(os, os_1):
    from __future__ import annotations
    _set_env('LANGCHAIN_API_KEY')
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_PROJECT'] = 'langchain-academy'
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
        ## Visibility into Trustcall updates

        Trustcall creates and updates JSON schemas.

        What if we want visibility into the *specific changes* made by Trustcall?

        For example, we saw before that Trustcall has some of its own tools to:

        * Self-correct from validation failures -- [see trace example here](https://smith.langchain.com/public/5cd23009-3e05-4b00-99f0-c66ee3edd06e/r/9684db76-2003-443b-9aa2-9a9dbc5498b7) 
        * Update existing documents -- [see trace example here](https://smith.langchain.com/public/f45bdaf0-6963-4c19-8ec9-f4b7fe0f68ad/r/760f90e1-a5dc-48f1-8c34-79d6a3414ac3)

        Visibility into these tools can be useful for the agent we're going to build.

        Below, we'll show how to do this!
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
def __(mo):
    mo.md(
        r"""
        We can add a [listener](https://python.langchain.com/docs/how_to/lcel_cheatsheet/#add-lifecycle-listeners) to the Trustcall extractor.

        This will pass runs from the extractor's execution to a class, `Spy`, that we will define.

        Our `Spy` class will extract information about what tool calls were made by Trustcall.
        """
    )
    return


@app.cell
def __(Memory, model):
    from trustcall import create_extractor
    from langchain_openai import ChatOpenAI

    class Spy:

        def __init__(self):
            self.called_tools = []

        def __call__(self, run):
            q = [run]
            while q:
                r = q.pop()
                if r.child_runs:
                    q.extend(r.child_runs)
                if r.run_type == 'chat_model':
                    self.called_tools.append(r.outputs['generations'][0][0]['message']['kwargs']['tool_calls'])
    spy = Spy()
    _model = ChatOpenAI(model='gpt-4o', temperature=0)
    trustcall_extractor = create_extractor(model, tools=[Memory], tool_choice='Memory', enable_inserts=True)
    trustcall_extractor_see_all_tool_calls = trustcall_extractor.with_listeners(on_end=spy)
    return (
        ChatOpenAI,
        Spy,
        create_extractor,
        spy,
        trustcall_extractor,
        trustcall_extractor_see_all_tool_calls,
    )


@app.cell
def __(trustcall_extractor):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    # Instruction
    instruction = """Extract memories from the following conversation:"""

    # Conversation
    conversation = [HumanMessage(content="Hi, I'm Lance."),
                    AIMessage(content="Nice to meet you, Lance."),
                    HumanMessage(content="This morning I had a nice bike ride in San Francisco.")]

    # Invoke the extractor
    result = trustcall_extractor.invoke({"messages": [SystemMessage(content=instruction)] + conversation})
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
def __(AIMessage, HumanMessage, result):
    # Update the conversation
    updated_conversation = [AIMessage(content="That's great, did you do after?"),
                            HumanMessage(content="I went to Tartine and ate a croissant."),
                            AIMessage(content="What else is on your mind?"),
                            HumanMessage(content="I was thinking about my Japan, and going back this winter!"),]

    # Update the instruction
    system_msg = """Update existing memories and create new ones based on the following conversation:"""

    # We'll save existing memories, giving them an ID, key (tool name), and value
    tool_name = "Memory"
    existing_memories = [(str(i), tool_name, memory.model_dump()) for i, memory in enumerate(result["responses"])] if result["responses"] else None
    existing_memories
    return existing_memories, system_msg, tool_name, updated_conversation


@app.cell
def __(
    existing_memories,
    trustcall_extractor_see_all_tool_calls,
    updated_conversation,
):
    result_1 = trustcall_extractor_see_all_tool_calls.invoke({'messages': updated_conversation, 'existing': existing_memories})
    return (result_1,)


@app.cell
def __(result_1):
    for _m in result_1['response_metadata']:
        print(_m)
    return


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
def __(spy):
    # Inspect the tool calls made by Trustcall
    spy.called_tools
    return


@app.cell
def __(spy):
    def extract_tool_info(tool_calls, schema_name="Memory"):
        """Extract information from tool calls for both patches and new memories.

        Args:
            tool_calls: List of tool calls from the model
            schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
        """

        # Initialize list of changes
        changes = []

        for call_group in tool_calls:
            for call in call_group:
                if call['name'] == 'PatchDoc':
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                elif call['name'] == schema_name:
                    changes.append({
                        'type': 'new',
                        'value': call['args']
                    })

        # Format results as a single string
        result_parts = []
        for change in changes:
            if change['type'] == 'update':
                result_parts.append(
                    f"Document {change['doc_id']} updated:\n"
                    f"Plan: {change['planned_edits']}\n"
                    f"Added content: {change['value']}"
                )
            else:
                result_parts.append(
                    f"New {schema_name} created:\n"
                    f"Content: {change['value']}"
                )

        return "\n\n".join(result_parts)

    # Inspect spy.called_tools to see exactly what happened during the extraction
    schema_name = "Memory"
    changes = extract_tool_info(spy.called_tools, schema_name)
    print(changes)
    return changes, extract_tool_info, schema_name


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Creating an agent

        There are many different [agent](https://langchain-ai.github.io/langgraph/concepts/high_level/) architectures to choose from.

        Here, we'll implement something simple, a [ReAct](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation) agent.

        This agent will be a helpful companion for creating and managing a ToDo list.

        This agent can make a decision to update three types of long-term memory: 

        (a) Create or update a user `profile` with general user information 

        (b) Add or update items in a ToDo list `collection`

        (c) Update its own `instructions` on how to update items to the ToDo list
        """
    )
    return


@app.cell
def __():
    from typing import TypedDict, Literal

    # Update memory tool
    class UpdateMemory(TypedDict):
        """ Decision on what memory type to update """
        update_type: Literal['user', 'todo', 'instructions']
    return Literal, TypedDict, UpdateMemory


@app.cell
def __():
    _set_env("OPENAI_API_KEY")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Graph definition 

        We add a simple router, `route_message`, that makes a binary decision to save memories.

        The memory collection updating is handled by `Trustcall` in the `write_memory` node, as before!
        """
    )
    return


@app.cell
def __(
    BaseModel,
    ChatOpenAI,
    Field,
    HumanMessage_1,
    Literal,
    Spy,
    SystemMessage_1,
    UpdateMemory,
    create_extractor,
    extract_tool_info,
    model,
):
    import uuid
    from IPython.display import Image, display
    from datetime import datetime
    from typing import Optional
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import merge_message_runs, HumanMessage, SystemMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, MessagesState, END, START
    from langgraph.store.base import BaseStore
    from langgraph.store.memory import InMemoryStore
    _model = ChatOpenAI(model='gpt-4o', temperature=0)

    class Profile(BaseModel):
        """This is the profile of the user you are chatting with"""
        name: Optional[str] = Field(description="The user's name", default=None)
        location: Optional[str] = Field(description="The user's location", default=None)
        job: Optional[str] = Field(description="The user's job", default=None)
        connections: list[str] = Field(description='Personal connection of the user, such as family members, friends, or coworkers', default_factory=list)
        interests: list[str] = Field(description='Interests that the user has', default_factory=list)

    class ToDo(BaseModel):
        task: str = Field(description='The task to be completed.')
        time_to_complete: Optional[int] = Field(description='Estimated time to complete the task (minutes).')
        deadline: Optional[datetime] = Field(description='When the task needs to be completed by (if applicable)', default=None)
        solutions: list[str] = Field(description='List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)', min_items=1, default_factory=list)
        status: Literal['not started', 'in progress', 'done', 'archived'] = Field(description='Current status of the task', default='not started')
    profile_extractor = create_extractor(model, tools=[Profile], tool_choice='Profile')
    MODEL_SYSTEM_MESSAGE = "You are a helpful chatbot.\n\nYou are designed to be a companion to a user, helping them keep track of their ToDo list.\n\nYou have a long term memory which keeps track of three things:\n1. The user's profile (general information about them)\n2. The user's ToDo list\n3. General instructions for updating the ToDo list\n\nHere is the current User Profile (may be empty if no information has been collected yet):\n<user_profile>\n{user_profile}\n</user_profile>\n\nHere is the current ToDo List (may be empty if no tasks have been added yet):\n<todo>\n{todo}\n</todo>\n\nHere are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):\n<instructions>\n{instructions}\n</instructions>\n\nHere are your instructions for reasoning about the user's messages:\n\n1. Reason carefully about the user's messages as presented below.\n\n2. Decide whether any of the your long-term memory should be updated:\n- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`\n- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`\n- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`\n\n3. Tell the user that you have updated your memory, if appropriate:\n- Do not tell the user you have updated the user's profile\n- Tell the user them when you update the todo list\n- Do not tell the user that you have updated instructions\n\n4. Err on the side of updating the todo list. No need to ask for explicit permission.\n\n5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."
    TRUSTCALL_INSTRUCTION = 'Reflect on following interaction.\n\nUse the provided tools to retain any necessary memories about the user.\n\nUse parallel tool calling to handle updates and insertions simultaneously.\n\nSystem Time: {time}'
    CREATE_INSTRUCTIONS = 'Reflect on the following interaction.\n\nBased on this interaction, update your instructions for how to update ToDo list items.\n\nUse any feedback from the user to update how they like to have items added, etc.\n\nYour current instructions are:\n\n<current_instructions>\n{current_instructions}\n</current_instructions>'

    def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Load memories from the store and use them to personalize the chatbot's response."""
        user_id = config['configurable']['user_id']
        namespace = ('profile', user_id)
        memories = store.search(namespace)
        if memories:
            user_profile = memories[0].value
        else:
            user_profile = None
        namespace = ('todo', user_id)
        memories = store.search(namespace)
        todo = '\n'.join((f'{mem.value}' for mem in memories))
        namespace = ('instructions', user_id)
        memories = store.search(namespace)
        if memories:
            instructions = memories[0].value
        else:
            instructions = ''
        system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo, instructions=instructions)
        response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke([SystemMessage_1(content=system_msg)] + state['messages'])
        return {'messages': [response]}

    def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection."""
        user_id = config['configurable']['user_id']
        namespace = ('profile', user_id)
        existing_items = store.search(namespace)
        tool_name = 'Profile'
        existing_memories = [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items] if existing_items else None
        TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages = list(merge_message_runs(messages=[SystemMessage_1(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state['messages'][:-1]))
        result = profile_extractor.invoke({'messages': updated_messages, 'existing': existing_memories})
        for r, rmeta in zip(result['responses'], result['response_metadata']):
            store.put(namespace, rmeta.get('json_doc_id', str(uuid.uuid4())), r.model_dump(mode='json'))
        tool_calls = state['messages'][-1].tool_calls
        return {'messages': [{'role': 'tool', 'content': 'updated profile', 'tool_call_id': tool_calls[0]['id']}]}

    def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection."""
        user_id = config['configurable']['user_id']
        namespace = ('todo', user_id)
        existing_items = store.search(namespace)
        tool_name = 'ToDo'
        existing_memories = [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items] if existing_items else None
        TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        updated_messages = list(merge_message_runs(messages=[SystemMessage_1(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state['messages'][:-1]))
        spy = Spy()
        todo_extractor = create_extractor(model, tools=[ToDo], tool_choice=tool_name, enable_inserts=True).with_listeners(on_end=spy)
        result = todo_extractor.invoke({'messages': updated_messages, 'existing': existing_memories})
        for r, rmeta in zip(result['responses'], result['response_metadata']):
            store.put(namespace, rmeta.get('json_doc_id', str(uuid.uuid4())), r.model_dump(mode='json'))
        tool_calls = state['messages'][-1].tool_calls
        todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
        return {'messages': [{'role': 'tool', 'content': todo_update_msg, 'tool_call_id': tool_calls[0]['id']}]}

    def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Reflect on the chat history and update the memory collection."""
        user_id = config['configurable']['user_id']
        namespace = ('instructions', user_id)
        existing_memory = store.get(namespace, 'user_instructions')
        system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
        new_memory = model.invoke([SystemMessage_1(content=system_msg)] + state['messages'][:-1] + [HumanMessage_1(content='Please update the instructions based on the conversation')])
        key = 'user_instructions'
        store.put(namespace, key, {'memory': new_memory.content})
        tool_calls = state['messages'][-1].tool_calls
        return {'messages': [{'role': 'tool', 'content': 'updated instructions', 'tool_call_id': tool_calls[0]['id']}]}

    def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, 'update_todos', 'update_instructions', 'update_profile']:
        """Reflect on the memories and chat history to decide whether to update the memory collection."""
        message = state['messages'][-1]
        if len(message.tool_calls) == 0:
            return END
        else:
            tool_call = message.tool_calls[0]
            if tool_call['args']['update_type'] == 'user':
                return 'update_profile'
            elif tool_call['args']['update_type'] == 'todo':
                return 'update_todos'
            elif tool_call['args']['update_type'] == 'instructions':
                return 'update_instructions'
            else:
                raise ValueError
    builder = StateGraph(MessagesState)
    builder.add_node(task_mAIstro)
    builder.add_node(update_todos)
    builder.add_node(update_profile)
    builder.add_node(update_instructions)
    builder.add_edge(START, 'task_mAIstro')
    builder.add_conditional_edges('task_mAIstro', route_message)
    builder.add_edge('update_todos', 'task_mAIstro')
    builder.add_edge('update_profile', 'task_mAIstro')
    builder.add_edge('update_instructions', 'task_mAIstro')
    across_thread_memory = InMemoryStore()
    within_thread_memory = MemorySaver()
    graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)
    display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    return (
        BaseStore,
        CREATE_INSTRUCTIONS,
        END,
        HumanMessage,
        Image,
        InMemoryStore,
        MODEL_SYSTEM_MESSAGE,
        MemorySaver,
        MessagesState,
        Optional,
        Profile,
        RunnableConfig,
        START,
        StateGraph,
        SystemMessage,
        TRUSTCALL_INSTRUCTION,
        ToDo,
        across_thread_memory,
        builder,
        datetime,
        display,
        graph,
        merge_message_runs,
        profile_extractor,
        route_message,
        task_mAIstro,
        update_instructions,
        update_profile,
        update_todos,
        uuid,
        within_thread_memory,
    )


@app.cell
def __(HumanMessage_1, graph):
    config = {'configurable': {'thread_id': '1', 'user_id': 'Lance'}}
    _input_messages = [HumanMessage_1(content='My name is Lance. I live in SF with my wife. I have a 1 year old daughter.')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return (config,)


@app.cell
def __(HumanMessage_1, config, graph):
    _input_messages = [HumanMessage_1(content='My wife asked me to book swim lessons for the baby.')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(HumanMessage_1, config, graph):
    _input_messages = [HumanMessage_1(content='When creating or updating ToDo items, include specific local businesses / vendors.')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(across_thread_memory):
    _user_id = 'Lance'
    for _memory in across_thread_memory.search(('instructions', _user_id)):
        print(_memory.value)
    return


@app.cell
def __(HumanMessage_1, config, graph):
    _input_messages = [HumanMessage_1(content='I need to fix the jammed electric Yale lock on the door.')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(across_thread_memory):
    _user_id = 'Lance'
    for _memory in across_thread_memory.search(('todo', _user_id)):
        print(_memory.value)
    return


@app.cell
def __(HumanMessage_1, config, graph):
    _input_messages = [HumanMessage_1(content='For the swim lessons, I need to get that done by end of November.')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see that Trustcall performs patching of the existing memory:

        https://smith.langchain.com/public/4ad3a8af-3b1e-493d-b163-3111aa3d575a/r
        """
    )
    return


@app.cell
def __(HumanMessage_1, config, graph):
    _input_messages = [HumanMessage_1(content='Need to call back City Toyota to schedule car service.')]
    for _chunk in graph.stream({'messages': _input_messages}, config, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(across_thread_memory):
    _user_id = 'Lance'
    for _memory in across_thread_memory.search(('todo', _user_id)):
        print(_memory.value)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now we can create a new thread.

        This creates a new session. 

        Profile, ToDos, and Instructions saved to long-term memory are accessed. 
        """
    )
    return


@app.cell
def __(HumanMessage_1, graph):
    config_1 = {'configurable': {'thread_id': '2', 'user_id': 'Lance'}}
    _input_messages = [HumanMessage_1(content='I have 30 minutes, what tasks can I get done?')]
    for _chunk in graph.stream({'messages': _input_messages}, config_1, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return (config_1,)


@app.cell
def __(HumanMessage_1, config_1, graph):
    _input_messages = [HumanMessage_1(content='Yes, give me some options to call for swim lessons.')]
    for _chunk in graph.stream({'messages': _input_messages}, config_1, stream_mode='values'):
        _chunk['messages'][-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Trace: 

        https://smith.langchain.com/public/84768705-be91-43e4-8a6f-f9d3cee93782/r

        ## Studio

        ![Screenshot 2024-11-04 at 1.00.19 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6732cfb05d9709862eba4e6c_Screenshot%202024-11-11%20at%207.46.40%E2%80%AFPM.png)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

