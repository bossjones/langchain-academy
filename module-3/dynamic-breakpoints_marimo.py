import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/dynamic-breakpoints.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239526-lesson-4-dynamic-breakpoints)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Dynamic breakpoints 

        ## Review

        We discussed motivations for human-in-the-loop:

        (1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action

        (2) `Debugging` - We can rewind the graph to reproduce or avoid issues

        (3) `Editing` - You can modify the state 

        We covered breakpoints as a general way to stop the graph at specific steps, which enables use-cases like `Approval`

        We also showed how to edit graph state, and introduce human feedback. 

        ## Goals

        Breakpoints are set by the developer on a specific node during graph compilation. 

        But, sometimes it is helpful to allow the graph **dynamically interrupt** itself!

        This is an internal breakpoint, and [can be achieved using `NodeInterrupt`](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/dynamic_breakpoints/#run-the-graph-with-dynamic-interrupt).

        This has a few specific benefits: 

        (1) you can do it conditionally (from inside a node based on developer-defined logic).

        (2) you can communicate to the user why its interrupted (by passing whatever you want to the `NodeInterrupt`).

        Let's create a graph where a `NodeInterrupt` is thrown based upon length of the input.
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
    from IPython.display import Image, display

    from typing_extensions import TypedDict
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.errors import NodeInterrupt
    from langgraph.graph import START, END, StateGraph

    class State(TypedDict):
        input: str

    def step_1(state: State) -> State:
        print("---Step 1---")
        return state

    def step_2(state: State) -> State:
        # Let's optionally raise a NodeInterrupt if the length of the input is longer than 5 characters
        if len(state['input']) > 5:
            raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
        
        print("---Step 2---")
        return state

    def step_3(state: State) -> State:
        print("---Step 3---")
        return state

    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    # Set up memory
    memory = MemorySaver()

    # Compile the graph with memory
    graph = builder.compile(checkpointer=memory)

    # View
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        END,
        Image,
        MemorySaver,
        NodeInterrupt,
        START,
        State,
        StateGraph,
        TypedDict,
        builder,
        display,
        graph,
        memory,
        step_1,
        step_2,
        step_3,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's run the graph with an input that's longer than 5 characters. 
        """
    )
    return


@app.cell
def __(graph):
    initial_input = {'input': 'hello world'}
    thread_config = {'configurable': {'thread_id': '1'}}
    for _event in graph.stream(initial_input, thread_config, stream_mode='values'):
        print(_event)
    return initial_input, thread_config


@app.cell
def __(mo):
    mo.md(
        r"""
        If we inspect the graph state at this point, we the node set to execute next (`step_2`).

        """
    )
    return


@app.cell
def __(graph, thread_config):
    state = graph.get_state(thread_config)
    print(state.next)
    return (state,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see that the `Interrupt` is logged to state.
        """
    )
    return


@app.cell
def __(state):
    print(state.tasks)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can try to resume the graph from the breakpoint. 

        But, this just re-runs the same node! 

        Unless state is changed we will be stuck here.
        """
    )
    return


@app.cell
def __(graph, thread_config):
    for _event in graph.stream(None, thread_config, stream_mode='values'):
        print(_event)
    return


@app.cell
def __(graph, thread_config):
    state_1 = graph.get_state(thread_config)
    print(state_1.next)
    return (state_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can update state.
        """
    )
    return


@app.cell
def __(graph, thread_config):
    graph.update_state(
        thread_config,
        {"input": "hi"},
    )
    return


@app.cell
def __(graph, thread_config):
    for _event in graph.stream(None, thread_config, stream_mode='values'):
        print(_event)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Usage with LangGraph API

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        We can run the above graph in Studio with `module-3/studio/dynamic_breakpoints.py`.

        ![Screenshot 2024-08-27 at 2.02.20 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbaedf43c3d4df239c589e_dynamic-breakpoints1.png)
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
def __(mo):
    mo.md(
        r"""
        We connect to it via the SDK.
        """
    )
    return


@app.cell
async def __():
    from langgraph_sdk import get_client

    # Replace this with the URL of your own deployed graph
    URL = "http://localhost:62575"
    client = get_client(url=URL)

    # Search all hosted graphs
    assistants = await client.assistants.search()
    return URL, assistants, client, get_client


@app.cell
async def __(client):
    thread = await client.threads.create()
    input_dict = {'input': 'hello world'}
    async for _chunk in client.runs.stream(thread['thread_id'], assistant_id='dynamic_breakpoints', input=input_dict, stream_mode='values'):
        print(f'Receiving new event of type: {_chunk.event}...')
        print(_chunk.data)
        print('\n\n')
    return input_dict, thread


@app.cell
async def __(client, thread):
    current_state = await client.threads.get_state(thread['thread_id'])
    return (current_state,)


@app.cell
def __(current_state):
    current_state['next']
    return


@app.cell
async def __(client, thread):
    await client.threads.update_state(thread['thread_id'], {"input": "hi!"})
    return


@app.cell
async def __(client, thread):
    async for _chunk in client.runs.stream(thread['thread_id'], assistant_id='dynamic_breakpoints', input=None, stream_mode='values'):
        print(f'Receiving new event of type: {_chunk.event}...')
        print(_chunk.data)
        print('\n\n')
    return


@app.cell
async def __(client, thread):
    current_state_1 = await client.threads.get_state(thread['thread_id'])
    current_state_1
    return (current_state_1,)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

