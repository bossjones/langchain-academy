import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239412-lesson-5-router)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Router

        ## Review

        We built a graph that uses `messages` as state and a chat model with bound tools.

        We saw that the graph can:

        * Return a tool call
        * Return a natural language response

        ## Goals

        We can think of this as a router, where the chat model routes between a direct response or a tool call based upon the user input.

        This is an simple example of an agent, where the LLM is directing the control flow either by calling a tool or just responding directly. 

        ![Screenshot 2024-08-21 at 9.24.09 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac6543c3d4df239a4ed1_router1.png)

        Let's extend our graph to work with either output! 

        For this, we can use two ideas:

        (1) Add a node that will call our tool.

        (2) Add a conditional edge that will look at the chat model model output, and route to our tool calling node or simply end if no tool call is performed. 


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
def __():
    from langchain_openai import ChatOpenAI

    def multiply(a: int, b: int) -> int:
        """Multiply a and b.

        Args:
            a: first int
            b: second int
        """
        return a * b

    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools([multiply])
    return ChatOpenAI, llm, llm_with_tools, multiply


@app.cell
def __(mo):
    mo.md(
        r"""
         We use the [built-in `ToolNode`](https://langchain-ai.github.io/langgraph/reference/prebuilt/?h=tools+condition#toolnode) and simply pass a list of our tools to initialize it. 
         
         We use the [built-in `tools_condition`](https://langchain-ai.github.io/langgraph/reference/prebuilt/?h=tools+condition#tools_condition) as our conditional edge.
        """
    )
    return


@app.cell
def __(llm_with_tools, multiply):
    from IPython.display import Image, display
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph import MessagesState
    from langgraph.prebuilt import ToolNode
    from langgraph.prebuilt import tools_condition

    # Node
    def tool_calling_llm(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode([multiply]))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", END)
    graph = builder.compile()

    # View
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        END,
        Image,
        MessagesState,
        START,
        StateGraph,
        ToolNode,
        builder,
        display,
        graph,
        tool_calling_llm,
        tools_condition,
    )


@app.cell
def __(graph):
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content="Hello world.")]
    messages = graph.invoke({"messages": messages})
    for m in messages['messages']:
        m.pretty_print()
    return HumanMessage, m, messages


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can see that the graph runs the tool!

        It responds with a `ToolMessage`. 

        ## LangGraph Studio

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        --

        Load the `router` in Studio, which uses `module-1/studio/router.py` set in `module-1/studio/langgraph.json`.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

