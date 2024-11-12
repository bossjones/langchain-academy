import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/multiple-schemas.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239434-lesson-3-multiple-schemas)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Multiple Schemas

        ## Review

        We just covered state schema and reducers.

        Typically, all graph nodes communicate with a single schema. 

        Also, this single schema contains the graph's input and output keys / channels.

        ## Goals

        But, there are cases where we may want a bit more control over this:

        * Internal nodes may pass information that is *not required* in the graph's input / output.

        * We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

        We'll discuss a few ways to customize graphs with multiple schemas.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install --quiet -U langgraph
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Private State

        First, let's cover the case of passing [private state](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/) between nodes.

        This is useful for anything needed as part of the intermediate working logic of the graph, but not relevant for the overall graph input or output.

        We'll define an `OverallState` and a `PrivateState`.

        `node_2` uses `PrivateState` as input, but writes out to `OverallState`.
        """
    )
    return


@app.cell
def __(OverallState):
    from typing_extensions import TypedDict
    from IPython.display import Image, display
    from langgraph.graph import StateGraph, START, END

    class _OverallState(TypedDict):
        foo: int

    class PrivateState(TypedDict):
        baz: int

    def node_1(state: OverallState) -> PrivateState:
        print('---Node 1---')
        return {'baz': state['foo'] + 1}

    def node_2(state: PrivateState) -> OverallState:
        print('---Node 2---')
        return {'foo': state['baz'] + 1}
    builder = StateGraph(OverallState)
    builder.add_node('node_1', node_1)
    builder.add_node('node_2', node_2)
    builder.add_edge(START, 'node_1')
    builder.add_edge('node_1', 'node_2')
    builder.add_edge('node_2', END)
    graph = builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        END,
        Image,
        PrivateState,
        START,
        StateGraph,
        TypedDict,
        builder,
        display,
        graph,
        node_1,
        node_2,
    )


@app.cell
def __(graph):
    graph.invoke({"foo" : 1})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        `baz` is only included in `PrivateState`.

        `node_2` uses `PrivateState` as input, but writes out to `OverallState`.

        So, we can see that `baz` is excluded from the graph output because it is not in `OverallState`.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Input / Output Schema

        By default, `StateGraph` takes in a single schema and all nodes are expected to communicate with that schema. 

        However, it is also possible to [define explicit input and output schemas for a graph](https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/?h=input+outp).

        Often, in these cases, we define an "internal" schema that contains *all* keys relevant to graph operations.

        But, we use specific `input` and `output` schemas to constrain the input and output.

        First, let's just run the graph with a single schema.
        """
    )
    return


@app.cell
def __(END, Image, OverallState, START, StateGraph, TypedDict, display):
    class _OverallState(TypedDict):
        question: str
        answer: str
        notes: str

    def _thinking_node(state: OverallState):
        return {'answer': 'bye', 'notes': '... his is name is Lance'}

    def _answer_node(state: OverallState):
        return {'answer': 'bye Lance'}
    graph_1 = StateGraph(OverallState)
    graph_1.add_node('answer_node', _answer_node)
    graph_1.add_node('thinking_node', _thinking_node)
    graph_1.add_edge(START, 'thinking_node')
    graph_1.add_edge('thinking_node', 'answer_node')
    graph_1.add_edge('answer_node', END)
    graph_1 = graph_1.compile()
    display(Image(graph_1.get_graph().draw_mermaid_png()))
    return (graph_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Notice that the output of invoke contains all keys in `OverallState`. 
        """
    )
    return


@app.cell
def __(graph_1):
    graph_1.invoke({'question': 'hi'})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's use a specific `input` and `output` schema with our graph.

        Here, `input` / `output` schemas perform *filtering* on what keys are permitted on the input and output of the graph. 

        In addition, we can use a type hint `state: InputState` to specify the input schema of each of our nodes.

        This is important when the graph is using multiple schemas.

        We use type hints below to, for example, show that the output of `answer_node` will be filtered to `OutputState`. 
        """
    )
    return


@app.cell
def __(END, Image, OverallState, START, StateGraph, TypedDict, display):
    class InputState(TypedDict):
        question: str

    class OutputState(TypedDict):
        answer: str

    class _OverallState(TypedDict):
        question: str
        answer: str
        notes: str

    def _thinking_node(state: InputState):
        return {'answer': 'bye', 'notes': '... his is name is Lance'}

    def _answer_node(state: OverallState) -> OutputState:
        return {'answer': 'bye Lance'}
    graph_2 = StateGraph(OverallState, input=InputState, output=OutputState)
    graph_2.add_node('answer_node', _answer_node)
    graph_2.add_node('thinking_node', _thinking_node)
    graph_2.add_edge(START, 'thinking_node')
    graph_2.add_edge('thinking_node', 'answer_node')
    graph_2.add_edge('answer_node', END)
    graph_2 = graph_2.compile()
    display(Image(graph_2.get_graph().draw_mermaid_png()))
    graph_2.invoke({'question': 'hi'})
    return InputState, OutputState, graph_2


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see the `output` schema constrains the output to only the `answer` key.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

