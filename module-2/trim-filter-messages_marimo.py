import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/trim-filter-messages.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239435-lesson-4-trim-and-filter-messages)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Filtering and trimming messages

        ## Review

        Now, we have a deeper understanding of a few things: 

        * How to customize the graph state schema
        * How to define custom state reducers
        * How to use multiple graph state schemas

        ## Goals

        Now, we can start using these concepts with models in LangGraph!
         
        In the next few sessions, we'll build towards a chatbot that has long-term memory.

        Because our chatbot will use messages, let's first talk a bit more about advanced ways to work with messages in graph state.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # # %pip install --quiet -U langchain_core langgraph langchain_openai
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

        We'll log to a project, `langchain-academy`. 
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
        ## Messages as state

        First, let's define some messages.
        """
    )
    return


@app.cell
def __():
    from pprint import pprint
    from langchain_core.messages import AIMessage, HumanMessage
    messages = [AIMessage(f'So you said you were researching ocean mammals?', name='Bot')]
    messages.append(HumanMessage(f'Yes, I know about whales. But what others should I learn about?', name='Lance'))
    for _m in messages:
        _m.pretty_print()
    return AIMessage, HumanMessage, messages, pprint


@app.cell
def __(mo):
    mo.md(
        r"""
        Recall we can pass them to a chat model.
        """
    )
    return


@app.cell
def __(messages):
    # pyright: reportMissingImports=false
    # pyright: reportMissingModuleSource=false
    # pylint: disable=bad-plugin-value
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o")
    llm.invoke(messages)
    return ChatOpenAI, llm


@app.cell
def __(mo):
    mo.md(
        r"""
        We can run our chat model in a simple graph with `MessagesState`.
        """
    )
    return


@app.cell
def __(builder, llm):
    from IPython.display import Image, display
    from langgraph.graph import MessagesState
    from langgraph.graph import StateGraph, START, END

    def _chat_model_node(state: MessagesState):
        return {'messages': llm.invoke(state['messages'])}
    _builder = StateGraph(MessagesState)
    _builder.add_node('chat_model', _chat_model_node)
    _builder.add_edge(START, 'chat_model')
    _builder.add_edge('chat_model', END)
    graph = builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return END, Image, MessagesState, START, StateGraph, display, graph


@app.cell
def __(graph, messages):
    output = graph.invoke({'messages': messages})
    for _m in output['messages']:
        _m.pretty_print()
    return (output,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Reducer

        A practical challenge when working with messages is managing long-running conversations. 

        Long-running conversations result in high token usage and latency if we are not careful, because we pass a growing list of messages to the model.

        We have a few ways to address this.

        First, recall the trick we saw using `RemoveMessage` and the `add_messages` reducer.
        """
    )
    return


@app.cell
def __(
    END,
    Image,
    MessagesState,
    START,
    StateGraph,
    builder,
    display,
    llm,
):
    from langchain_core.messages import RemoveMessage

    def filter_messages(state: MessagesState):
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
        return {'messages': delete_messages}

    def _chat_model_node(state: MessagesState):
        return {'messages': [llm.invoke(state['messages'])]}
    _builder = StateGraph(MessagesState)
    _builder.add_node('filter', filter_messages)
    _builder.add_node('chat_model', _chat_model_node)
    _builder.add_edge(START, 'filter')
    _builder.add_edge('filter', 'chat_model')
    _builder.add_edge('chat_model', END)
    graph_1 = builder.compile()
    display(Image(graph_1.get_graph().draw_mermaid_png()))
    return RemoveMessage, filter_messages, graph_1


@app.cell
def __(AIMessage, HumanMessage, graph_1):
    messages_1 = [AIMessage('Hi.', name='Bot', id='1')]
    messages_1.append(HumanMessage('Hi.', name='Lance', id='2'))
    messages_1.append(AIMessage('So you said you were researching ocean mammals?', name='Bot', id='3'))
    messages_1.append(HumanMessage('Yes, I know about whales. But what others should I learn about?', name='Lance', id='4'))
    output_1 = graph_1.invoke({'messages': messages_1})
    for _m in output_1['messages']:
        _m.pretty_print()
    return messages_1, output_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Filtering messages

        If you don't need or want to modify the graph state, you can just filter the messages you pass to the chat model.

        For example, just pass in a filtered list: `llm.invoke(messages[-1:])` to the model.
        """
    )
    return


@app.cell
def __(
    END,
    Image,
    MessagesState,
    START,
    StateGraph,
    builder,
    display,
    llm,
):
    def _chat_model_node(state: MessagesState):
        return {'messages': [llm.invoke(state['messages'][-1:])]}
    _builder = StateGraph(MessagesState)
    _builder.add_node('chat_model', _chat_model_node)
    _builder.add_edge(START, 'chat_model')
    _builder.add_edge('chat_model', END)
    graph_2 = builder.compile()
    display(Image(graph_2.get_graph().draw_mermaid_png()))
    return (graph_2,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's take our existing list of messages, append the above LLM response, and append a follow-up question.
        """
    )
    return


@app.cell
def __(HumanMessage, messages_1, output_1):
    messages_1.append(output_1['messages'][-1])
    messages_1.append(HumanMessage(f'Tell me more about Narwhals!', name='Lance'))
    return


@app.cell
def __(messages_1):
    for _m in messages_1:
        _m.pretty_print()
    return


@app.cell
def __(graph_2, messages_1):
    output_2 = graph_2.invoke({'messages': messages_1})
    for _m in output_2['messages']:
        _m.pretty_print()
    return (output_2,)


@app.cell
def __(mo):
    mo.md(
        r"""
        The state has all of the mesages.

        But, let's look at the LangSmith trace to see that the model invocation only uses the last message:

        https://smith.langchain.com/public/75aca3ce-ef19-4b92-94be-0178c7a660d9/r
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Trim messages

        Another approach is to [trim messages](https://python.langchain.com/v0.2/docs/how_to/trim_messages/#getting-the-last-max_tokens-tokens), based upon a set number of tokens. 

        This restricts the message history to a specified number of tokens.

        While filtering only returns a post-hoc subset of the messages between agents, trimming restricts the number of tokens that a chat model can use to respond.

        See the `trim_messages` below.
        """
    )
    return


@app.cell
def __(
    ChatOpenAI,
    END,
    Image,
    MessagesState,
    START,
    StateGraph,
    builder,
    display,
    llm,
):
    from langchain_core.messages import trim_messages

    def _chat_model_node(state: MessagesState):
        messages = trim_messages(state['messages'], max_tokens=100, strategy='last', token_counter=ChatOpenAI(model='gpt-4o'), allow_partial=False)
        return {'messages': [llm.invoke(messages)]}
    _builder = StateGraph(MessagesState)
    _builder.add_node('chat_model', _chat_model_node)
    _builder.add_edge(START, 'chat_model')
    _builder.add_edge('chat_model', END)
    graph_3 = builder.compile()
    display(Image(graph_3.get_graph().draw_mermaid_png()))
    return graph_3, trim_messages


@app.cell
def __(HumanMessage, messages_1, output_2):
    messages_1.append(output_2['messages'][-1])
    messages_1.append(HumanMessage(f'Tell me where Orcas live!', name='Lance'))
    return


@app.cell
def __(ChatOpenAI, messages_1, trim_messages):
    trim_messages(messages_1, max_tokens=100, strategy='last', token_counter=ChatOpenAI(model='gpt-4o'), allow_partial=False)
    return


@app.cell
def __(graph_3, messages_1):
    messages_out_trim = graph_3.invoke({'messages': messages_1})
    return (messages_out_trim,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's look at the LangSmith trace to see the model invocation:

        https://smith.langchain.com/public/b153f7e9-f1a5-4d60-8074-f0d7ab5b42ef/r
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

