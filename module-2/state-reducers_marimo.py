import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/state-reducers.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239428-lesson-2-state-reducers)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # State Reducers 

        ## Review

        We covered a few different ways to define LangGraph state schema, including `TypedDict`, `Pydantic`, or `Dataclasses`.
         
        ## Goals

        Now, we're going to dive into reducers, which specify how state updates are performed on specific keys / channels in the state schema.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # # %pip install --quiet -U langchain_core langgraph
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Default overwriting state

        Let's use a `TypedDict` as our state schema.
        """
    )
    return


@app.cell
def __(builder):
    from typing_extensions import TypedDict
    from IPython.display import Image, display
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        foo: int

    def node_1(state):
        print('---Node 1---')
        return {'foo': state['foo'] + 1}
    _builder = StateGraph(State)
    _builder.add_node('node_1', node_1)
    _builder.add_edge(START, 'node_1')
    _builder.add_edge('node_1', END)
    graph = builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        END,
        Image,
        START,
        State,
        StateGraph,
        TypedDict,
        display,
        graph,
        node_1,
    )


@app.cell
def __(graph):
    graph.invoke({"foo" : 1})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's look at the state update, `return {"foo": state['foo'] + 1}`.

        As discussed before, by default LangGraph doesn't know the preferred way to update the state.
         
        So, it will just overwrite the value of `foo` in `node_1`: 

        ```
        return {"foo": state['foo'] + 1}
        ```
         
        If we pass `{'foo': 1}` as input, the state returned from the graph is `{'foo': 2}`.

        ## Branching

        Let's look at a case where our nodes branch.
        """
    )
    return


@app.cell
def __(END, Image, START, StateGraph, TypedDict, builder, display):
    class State_1(TypedDict):
        foo: int

    def node_1_1(state):
        print('---Node 1---')
        return {'foo': state['foo'] + 1}

    def _node_2(state):
        print('---Node 2---')
        return {'foo': state['foo'] + 1}

    def _node_3(state):
        print('---Node 3---')
        return {'foo': state['foo'] + 1}
    _builder = StateGraph(State_1)
    _builder.add_node('node_1', node_1_1)
    _builder.add_node('node_2', _node_2)
    _builder.add_node('node_3', _node_3)
    _builder.add_edge(START, 'node_1')
    _builder.add_edge('node_1', 'node_2')
    _builder.add_edge('node_1', 'node_3')
    _builder.add_edge('node_2', END)
    _builder.add_edge('node_3', END)
    graph_1 = builder.compile()
    display(Image(graph_1.get_graph().draw_mermaid_png()))
    return State_1, graph_1, node_1_1


@app.cell
def __(graph_1):
    from langgraph.errors import InvalidUpdateError
    try:
        graph_1.invoke({'foo': 1})
    except InvalidUpdateError as e:
        print(f'InvalidUpdateError occurred: {e}')
    return (InvalidUpdateError,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We see a problem! 

        Node 1 branches to nodes 2 and 3.

        Nodes 2 and 3 run in parallel, which means they run in the same step of the graph.

        They both attempt to overwrite the state *within the same step*. 

        This is ambiguous for the graph! Which state should it keep? 
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Reducers

        [Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) give us a general way to address this problem.

        They specify how to perform updates.

        We can use the `Annotated` type to specify a reducer function. 

        For example, in this case let's append the value returned from each node rather than overwriting them.

        We just need a reducer that can perform this: `operator.add` is a function from Python's built-in operator module.

        When `operator.add` is applied to lists, it performs list concatenation.
        """
    )
    return


@app.cell
def __(END, Image, START, StateGraph, TypedDict, builder, display):
    from operator import add
    from typing import Annotated

    class State_2(TypedDict):
        foo: Annotated[list[int], add]

    def node_1_2(state):
        print('---Node 1---')
        return {'foo': [state['foo'][0] + 1]}
    _builder = StateGraph(State_2)
    _builder.add_node('node_1', node_1_2)
    _builder.add_edge(START, 'node_1')
    _builder.add_edge('node_1', END)
    graph_2 = builder.compile()
    display(Image(graph_2.get_graph().draw_mermaid_png()))
    return Annotated, State_2, add, graph_2, node_1_2


@app.cell
def __(graph_2):
    graph_2.invoke({'foo': [1]})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, our state key `foo` is a list.

        This `operator.add` reducer function will append updates from each node to this list. 
        """
    )
    return


@app.cell
def __(END, Image, START, StateGraph, State_2, builder, display):
    def node_1_3(state):
        print('---Node 1---')
        return {'foo': [state['foo'][-1] + 1]}

    def _node_2(state):
        print('---Node 2---')
        return {'foo': [state['foo'][-1] + 1]}

    def _node_3(state):
        print('---Node 3---')
        return {'foo': [state['foo'][-1] + 1]}
    _builder = StateGraph(State_2)
    _builder.add_node('node_1', node_1_3)
    _builder.add_node('node_2', _node_2)
    _builder.add_node('node_3', _node_3)
    _builder.add_edge(START, 'node_1')
    _builder.add_edge('node_1', 'node_2')
    _builder.add_edge('node_1', 'node_3')
    _builder.add_edge('node_2', END)
    _builder.add_edge('node_3', END)
    graph_3 = builder.compile()
    display(Image(graph_3.get_graph().draw_mermaid_png()))
    return graph_3, node_1_3


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see that updates in nodes 2 and 3 are performed concurrently because they are in the same step.
        """
    )
    return


@app.cell
def __(graph_3):
    graph_3.invoke({'foo': [1]})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's see what happens if we pass `None` to `foo`.

        We see an error because our reducer, `operator.add`, attempts to concatenate `NoneType` pass as input to list in `node_1`. 
        """
    )
    return


@app.cell
def __(graph_3):
    try:
        graph_3.invoke({'foo': None})
    except TypeError as e:
        print(f'TypeError occurred: {e}')
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Custom Reducers

        To address cases like this, [we can also define custom reducers](https://langchain-ai.github.io/langgraph/how-tos/subgraph/#custom-reducer-functions-to-manage-state). 

        For example, lets define custom reducer logic to combine lists and handle cases where either or both of the inputs might be `None`.
        """
    )
    return


@app.cell
def __(Annotated, TypedDict, add):
    def reduce_list(left: list | None, right: list | None) -> list:
        """Safely combine two lists, handling cases where either or both inputs might be None.

        Args:
            left (list | None): The first list to combine, or None.
            right (list | None): The second list to combine, or None.

        Returns:
            list: A new list containing all elements from both input lists.
                   If an input is None, it's treated as an empty list.
        """
        if not left:
            left = []
        if not right:
            right = []
        return left + right

    class DefaultState(TypedDict):
        foo: Annotated[list[int], add]

    class CustomReducerState(TypedDict):
        foo: Annotated[list[int], reduce_list]
    return CustomReducerState, DefaultState, reduce_list


@app.cell
def __(mo):
    mo.md(
        r"""
        In `node_1`, we append the value 2.
        """
    )
    return


@app.cell
def __(DefaultState, END, Image, START, StateGraph, builder, display):
    def node_1_4(state):
        print('---Node 1---')
        return {'foo': [2]}
    _builder = StateGraph(DefaultState)
    _builder.add_node('node_1', node_1_4)
    _builder.add_edge(START, 'node_1')
    _builder.add_edge('node_1', END)
    graph_4 = builder.compile()
    display(Image(graph_4.get_graph().draw_mermaid_png()))
    try:
        print(graph_4.invoke({'foo': None}))
    except TypeError as e:
        print(f'TypeError occurred: {e}')
    return graph_4, node_1_4


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, try with our custom reducer. We can see that no error is thrown.
        """
    )
    return


@app.cell
def __(
    CustomReducerState,
    END,
    Image,
    START,
    StateGraph,
    builder,
    display,
    node_1_4,
):
    _builder = StateGraph(CustomReducerState)
    _builder.add_node('node_1', node_1_4)
    _builder.add_edge(START, 'node_1')
    _builder.add_edge('node_1', END)
    graph_5 = builder.compile()
    display(Image(graph_5.get_graph().draw_mermaid_png()))
    try:
        print(graph_5.invoke({'foo': None}))
    except TypeError as e:
        print(f'TypeError occurred: {e}')
    return (graph_5,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Messages

        In module 1, we showed how to use a built-in reducer, `add_messages`, to handle messages in state.

        We also showed that [`MessagesState` is a useful shortcut if you want to work with messages](https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate). 

        * `MessagesState` has a built-in `messages` key 
        * It also has a built-in `add_messages` reducer for this key

        These two are equivalent. 

        We'll use the `MessagesState` class via `from langgraph.graph import MessagesState` for brevity.

        """
    )
    return


@app.cell
def __(Annotated, TypedDict):
    from langgraph.graph import MessagesState
    from langchain_core.messages import AnyMessage
    from langgraph.graph.message import add_messages

    # Define a custom TypedDict that includes a list of messages with add_messages reducer
    class CustomMessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        added_key_1: str
        added_key_2: str
        # etc

    # Use MessagesState, which includes the messages key with add_messages reducer
    class ExtendedMessagesState(MessagesState):
        # Add any keys needed beyond messages, which is pre-built
        added_key_1: str
        added_key_2: str
        # etc
    return (
        AnyMessage,
        CustomMessagesState,
        ExtendedMessagesState,
        MessagesState,
        add_messages,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's talk a bit more about usage of the `add_messages` reducer.
        """
    )
    return


@app.cell
def __(add_messages):
    from langchain_core.messages import AIMessage, HumanMessage
    _initial_messages = [AIMessage(content='Hello! How can I assist you?', name='Model'), HumanMessage(content="I'm looking for information on marine biology.", name='Lance')]
    _new_message = AIMessage(content='Sure, I can help with that. What specifically are you interested in?', name='Model')
    add_messages(_initial_messages, _new_message)
    return AIMessage, HumanMessage


@app.cell
def __(mo):
    mo.md(
        r"""
        So we can see that `add_messages` allows us to append messages to the `messages` key in our state.

        ### Re-writing

        Let's show some useful tricks when working with the `add_messages` reducer.

        If we pass a message with the same ID as an existing one in our `messages` list, it will get overwritten!
        """
    )
    return


@app.cell
def __(AIMessage, HumanMessage, add_messages):
    _initial_messages = [AIMessage(content='Hello! How can I assist you?', name='Model', id='1'), HumanMessage(content="I'm looking for information on marine biology.", name='Lance', id='2')]
    _new_message = HumanMessage(content="I'm looking for information on whales, specifically", name='Lance', id='2')
    add_messages(_initial_messages, _new_message)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Removal

        `add_messages` also [enables message removal](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/). 

        For this, we simply use [RemoveMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.modifier.RemoveMessage.html) from `langchain_core`.
        """
    )
    return


@app.cell
def __(AIMessage, HumanMessage):
    from langchain_core.messages import RemoveMessage

    # Message list
    messages = [AIMessage("Hi.", name="Bot", id="1")]
    messages.append(HumanMessage("Hi.", name="Lance", id="2"))
    messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
    messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

    # Isolate messages to delete
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
    print(delete_messages)
    return RemoveMessage, delete_messages, messages


@app.cell
def __(add_messages, delete_messages, messages):
    add_messages(messages , delete_messages)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see that mesage IDs 1 and 2, as noted in `delete_messages` are removed by the reducer.

        We'll see this put into practice a bit later.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

