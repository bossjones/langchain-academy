import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/edit-state-human-feedback.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239520-lesson-3-editing-state-and-human-feedback)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Editing graph state

        ## Review

        We discussed motivations for human-in-the-loop:

        (1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action

        (2) `Debugging` - We can rewind the graph to reproduce or avoid issues

        (3) `Editing` - You can modify the state 

        We showed how breakpoints support user approval, but don't yet know how to modify our graph state once our graph is interrupted!

        ## Goals

        Now, let's show how to directly edit the graph state and insert human feedback.
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
        ## Editing state 

        Previously, we introduced breakpoints.

        We used them to interrupt the graph and await user approval before executing the next node.

        But breakpoints are also [opportunities to modify the graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/).

        Let's set up our agent with a breakpoint before the `assistant` node.
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
def __(builder, llm_with_tools, memory, sys_msg, tools):
    from IPython.display import Image, display
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import MessagesState
    from langgraph.graph import START, StateGraph
    from langgraph.prebuilt import tools_condition, ToolNode
    from langchain_core.messages import HumanMessage, SystemMessage
    _sys_msg = SystemMessage(content='You are a helpful assistant tasked with performing arithmetic on a set of inputs.')

    def _assistant(state: MessagesState):
        return {'messages': [llm_with_tools.invoke([sys_msg] + state['messages'])]}
    _builder = StateGraph(MessagesState)
    _builder.add_node('assistant', _assistant)
    _builder.add_node('tools', ToolNode(tools))
    _builder.add_edge(START, 'assistant')
    _builder.add_conditional_edges('assistant', tools_condition)
    _builder.add_edge('tools', 'assistant')
    _memory = MemorySaver()
    graph = builder.compile(interrupt_before=['assistant'], checkpointer=memory)
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    return (
        HumanMessage,
        Image,
        MemorySaver,
        MessagesState,
        START,
        StateGraph,
        SystemMessage,
        ToolNode,
        display,
        graph,
        tools_condition,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's run!

        We can see the graph is interrupted before the chat model responds. 
        """
    )
    return


@app.cell
def __(graph):
    _initial_input = {'messages': 'Multiply 2 and 3'}
    thread = {'configurable': {'thread_id': '1'}}
    for _event in graph.stream(_initial_input, thread, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return (thread,)


@app.cell
def __(graph, thread):
    state = graph.get_state(thread)
    state
    return (state,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can directly apply a state update.

        Remember, updates to the `messages` key will use the `add_messages` reducer:
         
        * If we want to over-write the existing message, we can supply the message `id`.
        * If we simply want to append to our list of messages, then we can pass a message without an `id` specified, as shown below.
        """
    )
    return


@app.cell
def __(HumanMessage, graph, thread):
    graph.update_state(
        thread,
        {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's have a look.

        We called `update_state` with a new message. 

        The `add_messages` reducer appends it to our state key, `messages`.
        """
    )
    return


@app.cell
def __(graph, thread):
    new_state = graph.get_state(thread).values
    for m in new_state['messages']:
        m.pretty_print()
    return m, new_state


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's proceed with our agent, simply by passing `None` and allowing it proceed from the current state.

        We emit the current and then proceed to execute the remaining nodes.
        """
    )
    return


@app.cell
def __(graph, thread):
    for _event in graph.stream(None, thread, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we're back at the `assistant`, which has our `breakpoint`.

        We can again pass `None` to proceed.
        """
    )
    return


@app.cell
def __(graph, thread):
    for _event in graph.stream(None, thread, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Editing graph state in Studio

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        Let's load our `agent` in the Studio UI, which uses `module-3/studio/agent.py` set in `module-3/studio/langgraph.json`.

        ### Editing graph state with LangGraph API

        We can interact with our agent via the SDK.

        ![Screenshot 2024-08-26 at 9.59.19 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbaf2fbfb576f8e53ed930_edit-state-human-feedback1.png)

        Let's get the URL for the local deployment from Studio.

        The LangGraph API [supports editing graph state](https://langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_edit_state/#initial-invocation). 
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
    client = get_client(url="http://localhost:56091")
    return client, get_client


@app.cell
def __(mo):
    mo.md(
        r"""
        Our agent is defined in `assistant/agent.py`. 

        If you look at the code, you'll see that it *does not* have a breakpoint! 
         
        Of course, we can add it to `agent.py`, but one very nice feature of the API is that we can pass in a breakpoint!

        Here, we pass a `interrupt_before=["assistant"]`.
        """
    )
    return


@app.cell
async def __(chunk, client):
    _initial_input = {'messages': 'Multiply 2 and 3'}
    thread_1 = await client.threads.create()
    async for _chunk in client.runs.stream(thread_1['thread_id'], 'agent', input=_initial_input, stream_mode='values', interrupt_before=['assistant']):
        print(f'Receiving new event of type: {_chunk.event}...')
        _messages = chunk.data.get('messages', [])
        if _messages:
            print(_messages[-1])
        print('-' * 50)
    return (thread_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We can get the current state
        """
    )
    return


@app.cell
async def __(client, thread_1):
    current_state = await client.threads.get_state(thread_1['thread_id'])
    current_state
    return (current_state,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We can look at the last message in state.
        """
    )
    return


@app.cell
def __(current_state):
    last_message = current_state['values']['messages'][-1]
    last_message
    return (last_message,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We can edit it!
        """
    )
    return


@app.cell
def __(last_message):
    last_message['content'] = "No, actually multiply 3 and 3!"
    last_message
    return


@app.cell
def __(last_message):
    last_message
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Remember, as we said before, updates to the `messages` key will use the same `add_messages` reducer. 

        If we want to over-write the existing message, then we can supply the message `id`.

        Here, we did that. We only modified the message `content`, as shown above.
        """
    )
    return


@app.cell
async def __(client, last_message, thread_1):
    await client.threads.update_state(thread_1['thread_id'], {'messages': last_message})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we resume by passing `None`. 
        """
    )
    return


@app.cell
async def __(chunk, client, thread_1):
    async for _chunk in client.runs.stream(thread_1['thread_id'], assistant_id='agent', input=None, stream_mode='values', interrupt_before=['assistant']):
        print(f'Receiving new event of type: {_chunk.event}...')
        _messages = chunk.data.get('messages', [])
        if _messages:
            print(_messages[-1])
        print('-' * 50)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We get the result of the tool call as `9`, as expected.
        """
    )
    return


@app.cell
async def __(chunk, client, thread_1):
    async for _chunk in client.runs.stream(thread_1['thread_id'], assistant_id='agent', input=None, stream_mode='values', interrupt_before=['assistant']):
        print(f'Receiving new event of type: {_chunk.event}...')
        _messages = chunk.data.get('messages', [])
        if _messages:
            print(_messages[-1])
        print('-' * 50)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Awaiting user input

        So, it's clear that we can edit our agent state after a breakpoint.

        Now, what if we want to allow for human feedback to perform this state update?

        We'll add a node that [serves as a placeholder for human feedback](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/#setup) within our agent.

        This `human_feedback` node allow the user to add feedback directly to state.
         
        We specify the breakpoint using `interrupt_before` our `human_feedback` node.

        We set up a checkpointer to save the state of the graph up until this node.
        """
    )
    return


@app.cell
def __(
    Image,
    MemorySaver,
    MessagesState,
    START,
    StateGraph,
    SystemMessage,
    ToolNode,
    builder,
    display,
    llm_with_tools,
    memory,
    sys_msg,
    tools,
    tools_condition,
):
    _sys_msg = SystemMessage(content='You are a helpful assistant tasked with performing arithmetic on a set of inputs.')

    def human_feedback(state: MessagesState):
        pass

    def _assistant(state: MessagesState):
        return {'messages': [llm_with_tools.invoke([sys_msg] + state['messages'])]}
    _builder = StateGraph(MessagesState)
    _builder.add_node('assistant', _assistant)
    _builder.add_node('tools', ToolNode(tools))
    _builder.add_node('human_feedback', human_feedback)
    _builder.add_edge(START, 'human_feedback')
    _builder.add_edge('human_feedback', 'assistant')
    _builder.add_conditional_edges('assistant', tools_condition)
    _builder.add_edge('tools', 'human_feedback')
    _memory = MemorySaver()
    graph_1 = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
    display(Image(graph_1.get_graph().draw_mermaid_png()))
    return graph_1, human_feedback


@app.cell
def __(mo):
    mo.md(
        r"""
        We will get feedback from the user.

        We use `.update_state` to update the state of the graph with the human response we get, as before.

        We use the `as_node="human_feedback"` parameter to apply this state update as the specified node, `human_feedback`.
        """
    )
    return


@app.cell
def __(graph_1):
    _initial_input = {'messages': 'Multiply 2 and 3'}
    thread_2 = {'configurable': {'thread_id': '5'}}
    for _event in graph_1.stream(_initial_input, thread_2, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    user_input = input('Tell me how you want to update the state: ')
    graph_1.update_state(thread_2, {'messages': user_input}, as_node='human_feedback')
    for _event in graph_1.stream(None, thread_2, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return thread_2, user_input


@app.cell
def __(graph_1, thread_2):
    for _event in graph_1.stream(None, thread_2, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

