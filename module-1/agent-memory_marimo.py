import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239417-lesson-7-agent-with-memory)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Agent memory

        ## Review

        Previously, we built an agent that can:

        * `act` - let the model call specific tools 
        * `observe` - pass the tool output back to the model 
        * `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)

        ![Screenshot 2024-08-21 at 12.45.32 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab7453080e6802cd1703_agent-memory1.png)

        ## Goals

        Now, we're going extend our agent by introducing memory.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install --quiet -U langchain_openai langchain_core langgraph
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
        We'll use [LangSmith](https://docs.smith.langchain.com/) for [tracing](https://docs.smith.langchain.com/concepts/tracing).
        """
    )
    return


@app.cell
def __(os):
    _set_env("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        This follows what we did previously.
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
        """Divide a and b.

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
def __(llm_with_tools):
    from langgraph.graph import MessagesState
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    # Node
    def assistant(state: MessagesState):
       return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
    return (
        AIMessage,
        HumanMessage,
        MessagesState,
        SystemMessage,
        assistant,
        sys_msg,
    )


@app.cell
def __(MessagesState, assistant, tools):
    from langgraph.graph import START, StateGraph
    from langgraph.prebuilt import tools_condition, ToolNode
    from IPython.display import Image, display

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    react_graph = builder.compile()

    # Show
    display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
    return (
        Image,
        START,
        StateGraph,
        ToolNode,
        builder,
        display,
        react_graph,
        tools_condition,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Memory

        Let's run our agent, as before.
        """
    )
    return


@app.cell
def __(HumanMessage, messages, react_graph):
    _messages = [HumanMessage(content='Add 3 and 4.')]
    _messages = react_graph.invoke({'messages': messages})
    for _m in _messages['messages']:
        _m.pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's multiply by 2!
        """
    )
    return


@app.cell
def __(HumanMessage, messages, react_graph):
    _messages = [HumanMessage(content='Multiply that by 2.')]
    _messages = react_graph.invoke({'messages': messages})
    for _m in _messages['messages']:
        _m.pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We don't retain memory of 7 from our initial chat!

        This is because [state is transient](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) to a single graph execution.

        Of course, this limits our ability to have multi-turn conversations with interruptions. 

        We can use [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) to address this! 

        LangGraph can use a checkpointer to automatically save the graph state after each step.

        This built-in persistence layer gives us memory, allowing LangGraph to pick up from the last state update. 

        One of the easiest checkpointers to use is the `MemorySaver`, an in-memory key-value store for Graph state.

        All we need to do is simply compile the graph with a checkpointer, and our graph has memory!
        """
    )
    return


@app.cell
def __(builder):
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    react_graph_memory = builder.compile(checkpointer=memory)
    return MemorySaver, memory, react_graph_memory


@app.cell
def __(mo):
    mo.md(
        r"""
        When we use memory, we need to specify a `thread_id`.

        This `thread_id` will store our collection of graph states.

        Here is a cartoon:

        * The checkpointer write the state at every step of the graph
        * These checkpoints are saved in a thread 
        * We can access that thread in the future using the `thread_id`

        ![state.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e0e9f526b41a4ed9e2d28b_agent-memory2.png)

        """
    )
    return


@app.cell
def __(HumanMessage, messages, react_graph_memory):
    config = {'configurable': {'thread_id': '1'}}
    _messages = [HumanMessage(content='Add 3 and 4.')]
    _messages = react_graph_memory.invoke({'messages': messages}, config)
    for _m in _messages['messages']:
        _m.pretty_print()
    return (config,)


@app.cell
def __(mo):
    mo.md(
        r"""
        If we pass the same `thread_id`, then we can proceed from from the previously logged state checkpoint! 

        In this case, the above conversation is captured in the thread.

        The `HumanMessage` we pass (`"Multiply that by 2."`) is appended to the above conversation.

        So, the model now know that `that` refers to the `The sum of 3 and 4 is 7.`.
        """
    )
    return


@app.cell
def __(HumanMessage, config, messages, react_graph_memory):
    _messages = [HumanMessage(content='Multiply that by 2.')]
    _messages = react_graph_memory.invoke({'messages': messages}, config)
    for _m in _messages['messages']:
        _m.pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## LangGraph Studio

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        --

        Load the `agent` in the UI, which uses `module-1/studio/agent.py` set in `module-1/studio/langgraph.json`.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

