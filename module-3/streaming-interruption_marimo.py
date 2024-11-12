import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-3/streaming-interruption.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239464-lesson-1-streaming)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Streaming

        ## Review

        In module 2, covered a few ways to customize graph state and memory.
         
        We built up to a Chatbot with external memory that can sustain long-running conversations. 

        ## Goals

        This module will dive into `human-in-the-loop`, which builds on memory and allows users to interact directly with graphs in various ways. 

        To set the stage for `human-in-the-loop`, we'll first dive into streaming, which provides several ways to visualize graph output (e.g., node state or chat model tokens) over the course of execution.
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
def __(mo):
    mo.md(
        r"""
        ## Streaming

        LangGraph is built with [first class support for streaming](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming).

        Let's set up our Chatbot from Module 2, and show various way to stream outputs from the graph during execution. 
        """
    )
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
        Note that we use `RunnableConfig` with `call_model` to enable token-wise streaming. This is [only needed with python < 3.11](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/). We include in case you are running this notebook in CoLab, which will use python 3.x. 
        """
    )
    return


@app.cell
def __():
    from IPython.display import Image, display

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
    from langchain_core.runnables import RunnableConfig

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph import MessagesState

    # LLM
    model = ChatOpenAI(model="gpt-4o", temperature=0) 

    # State 
    class State(MessagesState):
        summary: str

    # Define the logic to call the model
    def call_model(state: State, config: RunnableConfig):
        
        # Get summary if it exists
        summary = state.get("summary", "")

        # If there is summary, then we add it
        if summary:
            
            # Add summary to system message
            system_message = f"Summary of conversation earlier: {summary}"

            # Append summary to any newer messages
            messages = [SystemMessage(content=system_message)] + state["messages"]
        
        else:
            messages = state["messages"]
        
        response = model.invoke(messages, config)
        return {"messages": response}

    def summarize_conversation(state: State):
        
        # First, we get any existing summary
        summary = state.get("summary", "")

        # Create our summarization prompt 
        if summary:
            
            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
            
        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = model.invoke(messages)
        
        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    # Determine whether to end or summarize the conversation
    def should_continue(state: State):
        
        """Return the next node to execute."""
        
        messages = state["messages"]
        
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"
        
        # Otherwise we can just end
        return END

    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        ChatOpenAI,
        END,
        HumanMessage,
        Image,
        MemorySaver,
        MessagesState,
        RemoveMessage,
        RunnableConfig,
        START,
        State,
        StateGraph,
        SystemMessage,
        call_model,
        display,
        graph,
        memory,
        model,
        should_continue,
        summarize_conversation,
        workflow,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Streaming full state

        Now, let's talk about ways to [stream our graph state](https://langchain-ai.github.io/langgraph/concepts/low_level/#streaming).

        `.stream` and `.astream` are sync and async methods for streaming back results. 
         
        LangGraph supports a few [different streaming modes](https://langchain-ai.github.io/langgraph/how-tos/stream-values/) for [graph state](https://langchain-ai.github.io/langgraph/how-tos/stream-values/):
         
        * `values`: This streams the full state of the graph after each node is called.
        * `updates`: This streams updates to the state of the graph after each node is called.

        ![values_vs_updates.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbaf892d24625a201744e5_streaming1.png)

        Let's look at `stream_mode="updates"`.

        Because we stream with `updates`, we only see updates to the state after node in the graph is run.

        Each `chunk` is a dict with `node_name` as the key and the updated state as the value.
        """
    )
    return


@app.cell
def __(HumanMessage, graph):
    config = {'configurable': {'thread_id': '1'}}
    for _chunk in graph.stream({'messages': [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode='updates'):
        print(_chunk)
    return (config,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's now just print the state update.
        """
    )
    return


@app.cell
def __(HumanMessage, config, graph):
    for _chunk in graph.stream({'messages': [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode='updates'):
        _chunk['conversation']['messages'].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can see `stream_mode="values"`.

        This is the `full state` of the graph after the `conversation` node is called.
        """
    )
    return


@app.cell
def __(HumanMessage, graph):
    config_1 = {'configurable': {'thread_id': '2'}}
    _input_message = HumanMessage(content="hi! I'm Lance")
    for _event in graph.stream({'messages': [_input_message]}, config_1, stream_mode='values'):
        for m in _event['messages']:
            m.pretty_print()
        print('---' * 25)
    return config_1, m


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Streaming tokens

        We often want to stream more than graph state.

        In particular, with chat model calls it is common to stream the tokens as they are generated.

        We can do this [using the `.astream_events` method](https://langchain-ai.github.io/langgraph/how-tos/streaming-from-final-node/#stream-outputs-from-the-final-node), which streams back events as they happen inside nodes!

        Each event is a dict with a few keys:
         
        * `event`: This is the type of event that is being emitted. 
        * `name`: This is the name of event.
        * `data`: This is the data associated with the event.
        * `metadata`: Contains`langgraph_node`, the node emitting the event.

        Let's have a look.
        """
    )
    return


@app.cell
async def __(HumanMessage, graph):
    config_2 = {'configurable': {'thread_id': '3'}}
    _input_message = HumanMessage(content='Tell me about the 49ers NFL team')
    async for _event in graph.astream_events({'messages': [_input_message]}, config_2, version='v2'):
        print(f'Node: {_event['metadata'].get('langgraph_node', '')}. Type: {_event['event']}. Name: {_event['name']}')
    return (config_2,)


@app.cell
def __(mo):
    mo.md(
        r"""
        The central point is that tokens from chat models within your graph have the `on_chat_model_stream` type.

        We can use `event['metadata']['langgraph_node']` to select the node to stream from.

        And we can use `event['data']` to get the actual data for each event, which in this case is an `AIMessageChunk`. 
        """
    )
    return


@app.cell
async def __(HumanMessage, graph):
    node_to_stream = 'conversation'
    config_3 = {'configurable': {'thread_id': '4'}}
    _input_message = HumanMessage(content='Tell me about the 49ers NFL team')
    async for _event in graph.astream_events({'messages': [_input_message]}, config_3, version='v2'):
        if _event['event'] == 'on_chat_model_stream' and _event['metadata'].get('langgraph_node', '') == node_to_stream:
            print(_event['data'])
    return config_3, node_to_stream


@app.cell
def __(mo):
    mo.md(
        r"""
        As you see above, just use the `chunk` key to get the `AIMessageChunk`.
        """
    )
    return


@app.cell
async def __(HumanMessage, event, graph, node_to_stream):
    config_4 = {'configurable': {'thread_id': '5'}}
    _input_message = HumanMessage(content='Tell me about the 49ers NFL team')
    async for _event in graph.astream_events({'messages': [_input_message]}, config_4, version='v2'):
        if _event['event'] == 'on_chat_model_stream' and _event['metadata'].get('langgraph_node', '') == node_to_stream:
            data = event['data']
            print(data['chunk'].content, end='|')
    return config_4, data


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Streaming with LangGraph API

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        The LangGraph API [has first class support for streaming](https://langchain-ai.github.io/langgraph/cloud/concepts/api/#streaming). 

        Let's load our `agent` in the Studio UI, which uses `module-3/studio/agent.py` set in `module-3/studio/langgraph.json`.

        The LangGraph API serves as the back-end for Studio.

        We can interact directly with the LangGraph API via the LangGraph SDK.

        We just need to get the URL for the local deployment from Studio.

        ![Screenshot 2024-08-27 at 2.20.34 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbaf8943c3d4df239cbf0f_streaming2.png)
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
async def __():
    from langgraph_sdk import get_client

    # Replace this with the URL of your own deployed graph
    URL = "http://localhost:56091"
    client = get_client(url=URL)

    # Search all hosted graphs
    assistants = await client.assistants.search()
    return URL, assistants, client, get_client


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's [stream `values`](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_values/), like before.
        """
    )
    return


@app.cell
async def __(HumanMessage, client):
    _thread = await client.threads.create()
    _input_message = HumanMessage(content='Multiply 2 and 3')
    async for _event in client.runs.stream(_thread['thread_id'], assistant_id='agent', input={'messages': [_input_message]}, stream_mode='values'):
        print(_event)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The streamed objects have: 

        * `event`: Type
        * `data`: State
        """
    )
    return


@app.cell
async def __(HumanMessage, client, event):
    from langchain_core.messages import convert_to_messages
    _thread = await client.threads.create()
    _input_message = HumanMessage(content='Multiply 2 and 3')
    async for _event in client.runs.stream(_thread['thread_id'], assistant_id='agent', input={'messages': [_input_message]}, stream_mode='values'):
        messages = event.data.get('messages', None)
        if messages:
            print(convert_to_messages(messages)[-1])
        print('=' * 25)
    return convert_to_messages, messages


@app.cell
def __(mo):
    mo.md(
        r"""
        There are some new streaming mode that are only supported via the API.

        For example, we can [use `messages` mode](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/) to better handle the above case!

        This mode currently assumes that you have a `messages` key in your graph, which is a list of messages.

        All events emitted using `messages` mode have two attributes:

        * `event`: This is the name of the event
        * `data`: This is data associated with the event
        """
    )
    return


@app.cell
async def __(HumanMessage, client):
    _thread = await client.threads.create()
    _input_message = HumanMessage(content='Multiply 2 and 3')
    async for _event in client.runs.stream(_thread['thread_id'], assistant_id='agent', input={'messages': [_input_message]}, stream_mode='messages'):
        print(_event.event)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see a few events: 

        * `metadata`: metadata about the run
        * `messages/complete`: fully formed message 
        * `messages/partial`: chat model tokens

        You can dig further into the types [here](https://langchain-ai.github.io/langgraph/cloud/concepts/api/#modemessages).

        Now, let's show how to stream these messages. 

        We'll define a helper function for better formatting of the tool calls in messages.
        """
    )
    return


@app.cell
async def __(HumanMessage, client):
    _thread = await client.threads.create()
    _input_message = HumanMessage(content='Multiply 2 and 3')

    def format_tool_calls(tool_calls):
        """
        Format a list of tool calls into a readable string.

        Args:
            tool_calls (list): A list of dictionaries, each representing a tool call.
                Each dictionary should have 'id', 'name', and 'args' keys.

        Returns:
            str: A formatted string of tool calls, or "No tool calls" if the list is empty.

        """
        if tool_calls:
            formatted_calls = []
            for call in tool_calls:
                formatted_calls.append(f'Tool Call ID: {call['id']}, Function: {call['name']}, Arguments: {call['args']}')
            return '\n'.join(formatted_calls)
        return 'No tool calls'
    async for _event in client.runs.stream(_thread['thread_id'], assistant_id='agent', input={'messages': [_input_message]}, stream_mode='messages'):
        if _event.event == 'metadata':
            print(f'Metadata: Run ID - {_event.data['run_id']}')
            print('-' * 50)
        elif _event.event == 'messages/partial':
            for data_item in _event.data:
                if 'role' in data_item and data_item['role'] == 'user':
                    print(f'Human: {data_item['content']}')
                else:
                    tool_calls = data_item.get('tool_calls', [])
                    invalid_tool_calls = data_item.get('invalid_tool_calls', [])
                    content = data_item.get('content', '')
                    response_metadata = data_item.get('response_metadata', {})
                    if content:
                        print(f'AI: {content}')
                    if tool_calls:
                        print('Tool Calls:')
                        print(format_tool_calls(tool_calls))
                    if invalid_tool_calls:
                        print('Invalid Tool Calls:')
                        print(format_tool_calls(invalid_tool_calls))
                    if response_metadata:
                        finish_reason = response_metadata.get('finish_reason', 'N/A')
                        print(f'Response Metadata: Finish Reason - {finish_reason}')
            print('-' * 50)
    return (
        content,
        data_item,
        finish_reason,
        format_tool_calls,
        invalid_tool_calls,
        response_metadata,
        tool_calls,
    )


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

