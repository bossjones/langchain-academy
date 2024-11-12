import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/time-travel.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239536-lesson-5-time-travel)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Time travel

        ## Review

        We discussed motivations for human-in-the-loop:

        (1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action

        (2) `Debugging` - We can rewind the graph to reproduce or avoid issues

        (3) `Editing` - You can modify the state 

        We showed how breakpoints can stop the graph at specific nodes or allow the graph to dynamically interrupt itself.

        Then we showed how to proceed with human approval or directly edit the graph state with human feedback.

        ## Goals

        Now, let's show how LangGraph [supports debugging](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/) by viewing, re-playing, and even forking from past states. 

        We call this `time travel`.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install --quiet -U langgraph langchain_openai langgraph_sdk
    return


@app.cell
def __():
    import os, getpass

    def _set_env(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"{var}: ")

    _set_env("OPENAI_API_KEY")
    return getpass, os


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's build our agent.
        """
    )
    return


@app.cell
def __():
    from langchain_openai import ChatOpenAI

    def multiply(a: int, b: int) -> int:
        """Multiply a and b.

        Args:
            a: first int
            b: second int
        """
        return a * b

    # This will be a tool
    def add(a: int, b: int) -> int:
        """Adds a and b.

        Args:
            a: first int
            b: second int
        """
        return a + b

    def divide(a: int, b: int) -> float:
        """Adds a and b.

        Args:
            a: first int
            b: second int
        """
        return a / b

    tools = [add, multiply, divide]
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)
    return ChatOpenAI, add, divide, llm, llm_with_tools, multiply, tools


@app.cell
def __(llm_with_tools, tools):
    from IPython.display import Image, display

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import MessagesState
    from langgraph.graph import START, END, StateGraph
    from langgraph.prebuilt import tools_condition, ToolNode

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    # Node
    def assistant(state: MessagesState):
       return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine the control flow
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=MemorySaver())

    # Show
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    return (
        AIMessage,
        END,
        HumanMessage,
        Image,
        MemorySaver,
        MessagesState,
        START,
        StateGraph,
        SystemMessage,
        ToolNode,
        assistant,
        builder,
        display,
        graph,
        memory,
        sys_msg,
        tools_condition,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's run it, as before.
        """
    )
    return


@app.cell
def __(HumanMessage, graph):
    _initial_input = {'messages': HumanMessage(content='Multiply 2 and 3')}
    thread = {'configurable': {'thread_id': '1'}}
    for _event in graph.stream(_initial_input, thread, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return (thread,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Browsing History

        We can use `get_state` to look at the **current** state of our graph, given the `thread_id`!
        """
    )
    return


@app.cell
def __(graph):
    graph.get_state({'configurable': {'thread_id': '1'}})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can also browse the state history of our agent.

        `get_state_history` lets us get the state at all prior steps.

        """
    )
    return


@app.cell
def __(graph, thread):
    all_states = [s for s in graph.get_state_history(thread)]
    return (all_states,)


@app.cell
def __(all_states):
    len(all_states)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The first element is the current state, just as we got from `get_state`.
        """
    )
    return


@app.cell
def __(all_states):
    all_states[-2]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Everything above we can visualize here: 

        ![fig1.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038211b544898570be3_time-travel1.png)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Replaying 

        We can re-run our agent from any of the prior steps.

        ![fig2.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038a0bd34b541c78fb8_time-travel2.png)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's look back at the step that recieved human input!
        """
    )
    return


@app.cell
def __(all_states):
    to_replay = all_states[-2]
    return (to_replay,)


@app.cell
def __(to_replay):
    to_replay
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Look at the state.
        """
    )
    return


@app.cell
def __(to_replay):
    to_replay.values
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see the next node to call.
        """
    )
    return


@app.cell
def __(to_replay):
    to_replay.next
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We also get the config, which tells us the `checkpoint_id` as well as the `thread_id`.
        """
    )
    return


@app.cell
def __(to_replay):
    to_replay.config
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        To replay from here, we simply pass the config back to the agent!

        The graph knows that this checkpoint has aleady been executed. 

        It just re-plays from this checkpoint!
        """
    )
    return


@app.cell
def __(graph, to_replay):
    for _event in graph.stream(None, to_replay.config, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can see our current state after the agent re-ran.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Forking

        What if we want to run from that same step, but with a different input.

        This is forking.

        ![fig3.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038f89f2d847ee5c336_time-travel3.png)
        """
    )
    return


@app.cell
def __(all_states):
    to_fork = all_states[-2]
    to_fork.values["messages"]
    return (to_fork,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Again, we have the config.
        """
    )
    return


@app.cell
def __(to_fork):
    to_fork.config
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's modify the state at this checkpoint.

        We can just run `update_state` with the `checkpoint_id` supplied. 

        Remember how our reducer on `messages` works: 

        * It will append, unless we supply a message ID.
        * We supply the message ID to overwrite the message, rather than appending to state!

        So, to overwrite the the message, we just supply the message ID, which we have `to_fork.values["messages"].id`.
        """
    )
    return


@app.cell
def __(HumanMessage, graph, to_fork):
    fork_config = graph.update_state(
        to_fork.config,
        {"messages": [HumanMessage(content='Multiply 5 and 3', 
                                   id=to_fork.values["messages"][0].id)]},
    )
    return (fork_config,)


@app.cell
def __(fork_config):
    fork_config
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        This creates a new, forked checkpoint.
         
        But, the metadata - e.g., where to go next - is perserved! 

        We can see the current state of our agent has been updated with our fork.
        """
    )
    return


@app.cell
def __(graph, thread):
    all_states_1 = [state for state in graph.get_state_history(thread)]
    all_states_1[0].values['messages']
    return (all_states_1,)


@app.cell
def __(graph):
    graph.get_state({'configurable': {'thread_id': '1'}})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, when we stream, the graph knows this checkpoint has never been executed.

        So, the graph runs, rather than simply re-playing.
        """
    )
    return


@app.cell
def __(fork_config, graph):
    for _event in graph.stream(None, fork_config, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can see the current state is the end of our agent run.
        """
    )
    return


@app.cell
def __(graph):
    graph.get_state({'configurable': {'thread_id': '1'}})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Time travel with LangGraph API

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        Let's load our `agent` in the Studio UI, which uses `module-3/studio/agent.py` set in `module-3/studio/langgraph.json`.

        ![Screenshot 2024-08-26 at 9.59.19 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb038211b544898570bec_time-travel4.png)

        We connect to it via the SDK and show how the LangGraph API [supports time travel](https://langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_time_travel/#initial-invocation). 
        """
    )
    return


@app.cell
def __(get_ipython):
    import platform

    if 'google.colab' in str(get_ipython()) or platform.system() != 'Darwin':
        raise Exception("Unfortunately LangGraph Studio is currently not supported on Google Colab or requires a Mac")
    return (platform,)


@app.cell
def __():
    from langgraph_sdk import get_client
    client = get_client(url="http://localhost:62780")
    return client, get_client


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Re-playing 

        Let's run our agent streaming `updates` to the state of the graph after each node is called.
        """
    )
    return


@app.cell
async def __(HumanMessage, chunk, client):
    _initial_input = {'messages': HumanMessage(content='Multiply 2 and 3')}
    thread_1 = await client.threads.create()
    async for _chunk in client.runs.stream(thread_1['thread_id'], assistant_id='agent', input=_initial_input, stream_mode='updates'):
        if _chunk.data:
            _assisant_node = chunk.data.get('assistant', {}).get('messages', [])
            _tool_node = chunk.data.get('tools', {}).get('messages', [])
            if _assisant_node:
                print('-' * 20 + 'Assistant Node' + '-' * 20)
                print(_assisant_node[-1])
            elif _tool_node:
                print('-' * 20 + 'Tools Node' + '-' * 20)
                print(_tool_node[-1])
    return (thread_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's look at **replaying** from a specified checkpoint. 

        We simply need to pass the `checkpoint_id`.
        """
    )
    return


@app.cell
async def __(client, states, thread_1):
    _states = await client.threads.get_history(thread_1['thread_id'])
    to_replay_1 = states[-2]
    to_replay_1
    return (to_replay_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's stream with `stream_mode="values"` to see the full state at every node as we replay. 
        """
    )
    return


@app.cell
async def __(client, thread_1, to_replay_1):
    async for _chunk in client.runs.stream(thread_1['thread_id'], assistant_id='agent', input=None, stream_mode='values', checkpoint_id=to_replay_1['checkpoint_id']):
        print(f'Receiving new event of type: {_chunk.event}...')
        print(_chunk.data)
        print('\n\n')
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can all view this as streaming only `updates` to state made by the nodes that we reply.
        """
    )
    return


@app.cell
async def __(chunk, client, thread_1, to_replay_1):
    async for _chunk in client.runs.stream(thread_1['thread_id'], assistant_id='agent', input=None, stream_mode='updates', checkpoint_id=to_replay_1['checkpoint_id']):
        if _chunk.data:
            _assisant_node = chunk.data.get('assistant', {}).get('messages', [])
            _tool_node = chunk.data.get('tools', {}).get('messages', [])
            if _assisant_node:
                print('-' * 20 + 'Assistant Node' + '-' * 20)
                print(_assisant_node[-1])
            elif _tool_node:
                print('-' * 20 + 'Tools Node' + '-' * 20)
                print(_tool_node[-1])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        #### Forking

        Now, let's look at forking.

        Let's get the same step as we worked with above, the human input.

        Let's create a new thread with out agent.
        """
    )
    return


@app.cell
async def __(HumanMessage, chunk, client):
    _initial_input = {'messages': HumanMessage(content='Multiply 2 and 3')}
    thread_2 = await client.threads.create()
    async for _chunk in client.runs.stream(thread_2['thread_id'], assistant_id='agent', input=_initial_input, stream_mode='updates'):
        if _chunk.data:
            _assisant_node = chunk.data.get('assistant', {}).get('messages', [])
            _tool_node = chunk.data.get('tools', {}).get('messages', [])
            if _assisant_node:
                print('-' * 20 + 'Assistant Node' + '-' * 20)
                print(_assisant_node[-1])
            elif _tool_node:
                print('-' * 20 + 'Tools Node' + '-' * 20)
                print(_tool_node[-1])
    return (thread_2,)


@app.cell
async def __(client, states, thread_2):
    _states = await client.threads.get_history(thread_2['thread_id'])
    to_fork_1 = states[-2]
    to_fork_1['values']
    return (to_fork_1,)


@app.cell
def __(to_fork_1):
    to_fork_1['values']['messages'][0]['id']
    return


@app.cell
def __(to_fork_1):
    to_fork_1['next']
    return


@app.cell
def __(to_fork_1):
    to_fork_1['checkpoint_id']
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's edit the state.

        Remember how our reducer on `messages` works: 

        * It will append, unless we supply a message ID.
        * We supply the message ID to overwrite the message, rather than appending to state!
        """
    )
    return


@app.cell
async def __(HumanMessage, client, thread_2, to_fork_1):
    forked_input = {'messages': HumanMessage(content='Multiply 3 and 3', id=to_fork_1['values']['messages'][0]['id'])}
    forked_config = await client.threads.update_state(thread_2['thread_id'], forked_input, checkpoint_id=to_fork_1['checkpoint_id'])
    return forked_config, forked_input


@app.cell
def __(forked_config):
    forked_config
    return


@app.cell
async def __(client, thread_2):
    _states = await client.threads.get_history(thread_2['thread_id'])
    _states[0]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        To rerun, we pass in the `checkpoint_id`.
        """
    )
    return


@app.cell
async def __(chunk, client, forked_config, thread_2):
    async for _chunk in client.runs.stream(thread_2['thread_id'], assistant_id='agent', input=None, stream_mode='updates', checkpoint_id=forked_config['checkpoint_id']):
        if _chunk.data:
            _assisant_node = chunk.data.get('assistant', {}).get('messages', [])
            _tool_node = chunk.data.get('tools', {}).get('messages', [])
            if _assisant_node:
                print('-' * 20 + 'Assistant Node' + '-' * 20)
                print(_assisant_node[-1])
            elif _tool_node:
                print('-' * 20 + 'Tools Node' + '-' * 20)
                print(_tool_node[-1])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### LangGraph Studio

        Let's look at forking in the Studio UI with our `agent`, which uses `module-1/studio/agent.py` set in `module-1/studio/langgraph.json`.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

