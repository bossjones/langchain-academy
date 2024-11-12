import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/breakpoints.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239469-lesson-2-breakpoints)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Breakpoints

        ## Review

        For `human-in-the-loop`, we often want to see our graph outputs as its running. 

        We laid the foundations for this with streaming. 

        ## Goals

        Now, let's talk about the motivations for `human-in-the-loop`:

        (1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action

        (2) `Debugging` - We can rewind the graph to reproduce or avoid issues

        (3) `Editing` - You can modify the state 

        LangGraph offers several ways to get or update agent state to support various `human-in-the-loop` workflows.

        First, we'll introduce [breakpoints](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/#simple-usage), which provide a simple way to stop the graph at specific steps. 

        We'll show how this enables user `approval`.
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
        ## Breakpoints for human approval

        Let's re-consider the simple agent that we worked with in Module 1. 

        Let's assume that are concerned about tool use: we want to approve the agent to use any of its tools.
         
        All we need to do is simply compile the graph with `interrupt_before=["tools"]` where `tools` is our tools node.

        This means that the execution will be interrupted before the node `tools`, which executes the tool call.
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
    from langgraph.graph import START, StateGraph
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
    graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)

    # Show
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    return (
        AIMessage,
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
        We can get the state and look at the next node to call.

        This is a nice way to see that the graph has been interrupted.
        """
    )
    return


@app.cell
def __(graph, thread):
    state = graph.get_state(thread)
    state.next
    return (state,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we'll introduce a nice trick.

        When we invoke the graph with `None`, it will just continue from the last state checkpoint!

        ![breakpoints.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbae7985b747dfed67775d_breakpoints1.png)

        For clarity, LangGraph will re-emit the current state, which contains the `AIMessage` with tool call.

        And then it will proceed to execute the following steps in the graph, which start with the tool node.

        We see that the tool node is run with this tool call, and it's passed back to the chat model for our final answer.
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
        Now, lets bring these together with a specific user approval step that accepts user input.
        """
    )
    return


@app.cell
def __(HumanMessage, graph):
    _initial_input = {'messages': HumanMessage(content='Multiply 2 and 3')}
    thread_1 = {'configurable': {'thread_id': '2'}}
    for _event in graph.stream(_initial_input, thread_1, stream_mode='values'):
        _event['messages'][-1].pretty_print()
    user_approval = input('Do you want to call the tool? (yes/no): ')
    if user_approval.lower() == 'yes':
        for _event in graph.stream(None, thread_1, stream_mode='values'):
            _event['messages'][-1].pretty_print()
    else:
        print('Operation cancelled by user.')
    return thread_1, user_approval


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Breakpoints with LangGraph API

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        Let's load our `agent` in the Studio UI, which uses `module-3/studio/agent.py` set in `module-3/studio/langgraph.json`.
         
        Let's get the URL for the local deployment from Studio.

        ![Screenshot 2024-08-26 at 9.36.41 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbae7989b1d60204c199dc_breakpoints2.png)

        The LangGraph API [supports breakpoints](https://langchain-ai.github.io/langgraph/cloud/how-tos/human_in_the_loop_breakpoint/#sdk-initialization). 
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
        As shown above, we can add `interrupt_before=["node"]` when compiling the graph that is running in Studio.

        However, with the API, you can also pass `interrupt_before` to the stream method directly. 
        """
    )
    return


@app.cell
async def __(HumanMessage, chunk, client):
    _initial_input = {'messages': HumanMessage(content='Multiply 2 and 3')}
    thread_2 = await client.threads.create()
    async for _chunk in client.runs.stream(thread_2['thread_id'], assistant_id='agent', input=_initial_input, stream_mode='values', interrupt_before=['tools']):
        print(f'Receiving new event of type: {_chunk.event}...')
        _messages = chunk.data.get('messages', [])
        if _messages:
            print(_messages[-1])
        print('-' * 50)
    return (thread_2,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can proceed from the breakpoint just like we did before by passing the `thread_id` and `None` as the input!
        """
    )
    return


@app.cell
async def __(chunk, client, thread_2):
    async for _chunk in client.runs.stream(thread_2['thread_id'], 'agent', input=None, stream_mode='values', interrupt_before=['tools']):
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

        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

