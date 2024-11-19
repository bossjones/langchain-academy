import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Connecting to a LangGraph Platform Deployment

        ## Deployment Creation

        We just created a [deployment](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph) for the `task_maistro` app from module 5.

        * We used the [the LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/#commands) to build a Docker image for the LangGraph Server with our `task_maistro` graph.
        * We used the provided `docker-compose.yml` file to create three separate containers based on the services defined: 
            * `langgraph-redis`: Creates a new container using the official Redis image.
            * `langgraph-postgres`: Creates a new container using the official Postgres image.
            * `langgraph-api`: Creates a new container using our pre-built `task_maistro` Docker image.

        ```
        $ cd module-6/deployment
        $ docker compose up
        ```

        Once running, we can access the deployment through:
              
        * API: http://localhost:8123
        * Docs: http://localhost:8123/docs
        * LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

        ![langgraph-platform-high-level.png](attachment:3a5ede4f-7a62-4e05-9e44-301465ca8555.png)

        ## Using the API  

        LangGraph Server exposes [many API endpoints](https://github.com/langchain-ai/agent-protocol) for interacting with the deployed agent.

        We can group [these endpoints into a few common agent needs](https://github.com/langchain-ai/agent-protocol): 

        * **Runs**: Atomic agent executions
        * **Threads**: Multi-turn interactions or human in the loop
        * **Store**: Long-term memory

        We can test requests directly [in the API docs](http://localhost:8123/docs#tag/thread-runs).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## SDK

        The [LangGraph SDKs](https://langchain-ai.github.io/langgraph/concepts/sdk/) (Python and JS) provide a developer-friendly interface to interact with the LangGraph Server API presented above.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # # %pip install -U langgraph_sdk
    return


@app.cell
def __(os):
    from __future__ import annotations

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"


    import os

    from dotenv import load_dotenv


    # Load environment variables from .env file for API access
    load_dotenv(dotenv_path=".env", override=True)

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    return annotations, load_dotenv, os


@app.cell
def __():
    from langgraph_sdk import get_client

    # Connect via SDK
    url_for_cli_deployment = "http://localhost:8123"
    client = get_client(url=url_for_cli_deployment)
    return client, get_client, url_for_cli_deployment


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Remote Graph

        If you are working in the LangGraph library, [Remote Graph](https://langchain-ai.github.io/langgraph/how-tos/use-remote-graph/) is also a useful way to connect directly to the graph.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install -U langchain_openai langgraph langchain_core
    return


@app.cell
def __():
    from langgraph.pregel.remote import RemoteGraph
    from langchain_core.messages import convert_to_messages
    from langchain_core.messages import HumanMessage, SystemMessage

    # Connect via remote graph
    url = "http://localhost:8123"
    graph_name = "task_maistro"
    remote_graph = RemoteGraph(graph_name, url=url)
    return (
        HumanMessage,
        RemoteGraph,
        SystemMessage,
        convert_to_messages,
        graph_name,
        remote_graph,
        url,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Runs

        A "run" represents a [single execution](https://github.com/langchain-ai/agent-protocol?tab=readme-ov-file#runs-atomic-agent-executions) of your graph. Each time a client makes a request:

        1. The HTTP worker generates a unique run ID
        2. This run and its results are stored in PostgreSQL
        3. You can query these runs to:
           - Check their status
           - Get their results
           - Track execution history

        You can see a full set of How To guides for various types of runs [here](https://langchain-ai.github.io/langgraph/how-tos/#runs).

        Let's looks at a few of the interesting things we can do with runs.

        ### Background Runs

        The LangGraph server supports two types of runs: 

        * `Fire and forget` - Launch a run in the background, but don’t wait for it to finish
        * `Waiting on a reply (blocking or polling)` - Launch a run and wait/stream its output

        Background runs and polling are quite useful when working with long-running agents. 

        Let's [see](https://langchain-ai.github.io/langgraph/cloud/how-tos/background_run/#check-runs-on-thread) how this works:
        """
    )
    return


@app.cell
async def __(client):
    # Create a thread
    thread = await client.threads.create()
    thread
    return (thread,)


@app.cell
async def __(client):
    thread_1 = await client.threads.create()
    runs = await client.runs.list(thread_1['thread_id'])
    print(runs)
    return runs, thread_1


@app.cell
async def __(HumanMessage, client, thread_1, user_input):
    _user_input = 'Add a ToDo to finish booking travel to Hong Kong by end of next week. Also, add a ToDo to call parents back about Thanksgiving plans.'
    config = {'configurable': {'user_id': 'Test'}}
    graph_name_1 = 'task_maistro'
    run = await client.runs.create(thread_1['thread_id'], graph_name_1, input={'messages': [HumanMessage(content=user_input)]}, config=config)
    return config, graph_name_1, run


@app.cell
async def __(HumanMessage, client, user_input):
    thread_2 = await client.threads.create()
    _user_input = 'Give me a summary of all ToDos.'
    config_1 = {'configurable': {'user_id': 'Test'}}
    graph_name_2 = 'task_maistro'
    run_1 = await client.runs.create(thread_2['thread_id'], graph_name_2, input={'messages': [HumanMessage(content=user_input)]}, config=config_1)
    return config_1, graph_name_2, run_1, thread_2


@app.cell
async def __(client, run_1, thread_2):
    print(await client.runs.get(thread_2['thread_id'], run_1['run_id']))
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see that it has `'status': 'pending'` because it is still running.

        What if we want to wait until the run completes, making it a blocking run?

        We can use `client.runs.join` to wait until the run completes.

        This ensures that no new runs are started until the current run completes on the thread.
        """
    )
    return


@app.cell
async def __(client, run_1, thread_2):
    await client.runs.join(thread_2['thread_id'], run_1['run_id'])
    print(await client.runs.get(thread_2['thread_id'], run_1['run_id']))
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now the run has `'status': 'success'` because it has completed.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Streaming Runs

        Each time a client makes a streaming request:

        1. The HTTP worker generates a unique run ID
        2. The Queue worker begins work on the run 
        3. During execution, the Queue worker publishes update to Redis
        4. The HTTP worker subscribes to updates from Redis for ths run, and returns them to the client 

        This enabled streaming! 

        We've covered [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming_1) in previous modules, but let's pick one method -- streaming tokens -- to highlight.

        Streaming tokens back to the client is especially useful when working with production agents that may take a while to complete.

        We [stream tokens](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/#setup) using `stream_mode="messages-tuple"`.
        """
    )
    return


@app.cell
async def __(HumanMessage, client, config_1, graph_name_2, thread_2):
    _user_input = 'What ToDo should I focus on first.'
    async for _chunk in client.runs.stream(thread_2['thread_id'], graph_name_2, input={'messages': [HumanMessage(content=_user_input)]}, config=config_1, stream_mode='messages-tuple'):
        if _chunk.event == 'messages':
            print(''.join((data_item['content'] for data_item in _chunk.data if 'content' in data_item)), end='', flush=True)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Threads

        Whereas a run is only a single execution of the graph, a thread supports *multi-turn* interactions.

        When the client makes a graph execution execution with a `thread_id`, the server will save all [checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints) (steps) in the run to the thread in the Postgres database.

        The server allows us to [check the status of created threads](https://langchain-ai.github.io/langgraph/cloud/how-tos/check_thread_status/).

        ### Check thread state

        In addition, we can easily access the state [checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints) saved to any specific thread.
        """
    )
    return


@app.cell
async def __(client, convert_to_messages, thread_2):
    thread_state = await client.threads.get_state(thread_2['thread_id'])
    for _m in convert_to_messages(thread_state['values']['messages']):
        _m.pretty_print()
    return (thread_state,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Copy threads

        We can also [copy](https://langchain-ai.github.io/langgraph/cloud/how-tos/copy_threads/) (i.e. "fork") an existing thread. 

        This will keep the existing thread's history, but allow us to create independent runs that do not affect the original thread.
        """
    )
    return


@app.cell
async def __(client, thread_2):
    copied_thread = await client.threads.copy(thread_2['thread_id'])
    return (copied_thread,)


@app.cell
async def __(client, convert_to_messages, copied_thread):
    copied_thread_state = await client.threads.get_state(copied_thread['thread_id'])
    for _m in convert_to_messages(copied_thread_state['values']['messages']):
        _m.pretty_print()
    return (copied_thread_state,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Human in the loop

        We covered [Human in the loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/) in Module 3, and the server supports all Human in the loop features that we discussed.

        As an example, [we can search, edit, and continue graph execution](https://langchain-ai.github.io/langgraph/concepts/persistence/#capabilities) from any prior checkpoint. 
        """
    )
    return


@app.cell
async def __(client, thread_2):
    states = await client.threads.get_history(thread_2['thread_id'])
    to_fork = states[-2]
    to_fork['values']
    return states, to_fork


@app.cell
def __(to_fork):
    to_fork['values']['messages'][0]['id']
    return


@app.cell
def __(to_fork):
    to_fork['next']
    return


@app.cell
def __(to_fork):
    to_fork['checkpoint_id']
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's edit the state. Remember how our reducer on `messages` works: 

        * It will append, unless we supply a message ID.
        * We supply the message ID to overwrite the message, rather than appending to state!
        """
    )
    return


@app.cell
async def __(HumanMessage, client, thread_2, to_fork):
    forked_input = {'messages': HumanMessage(content='Give me a summary of all ToDos that need to be done in the next week.', id=to_fork['values']['messages'][0]['id'])}
    forked_config = await client.threads.update_state(thread_2['thread_id'], forked_input, checkpoint_id=to_fork['checkpoint_id'])
    return forked_config, forked_input


@app.cell
async def __(client, config_1, forked_config, graph_name_2, thread_2):
    async for _chunk in client.runs.stream(thread_2['thread_id'], graph_name_2, input=None, config=config_1, checkpoint_id=forked_config['checkpoint_id'], stream_mode='messages-tuple'):
        if _chunk.event == 'messages':
            print(''.join((data_item['content'] for data_item in _chunk.data if 'content' in data_item)), end='', flush=True)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Across-thread memory

        In module 5, we covered how the [LangGraph memory `store`](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store) can be used to save information across threads.

        Our deployed graph, `task_maistro`, uses the `store` to save information -- such as ToDos -- namespaced to the `user_id`.

        Our deployment includes a Postgres database, which stores these long-term (across-thread) memories.

        There are several methods available [for interacting with the store](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient) in our deployment using the LangGraph SDK.

        ### Search items

        The `task_maistro` graph uses the `store` to save ToDos namespaced by default to (`todo`, `todo_category`, `user_id`). 

        The `todo_category` is by default set to `general` (as you can see in `deployment/configuration.py`).

        We can simply supply this tuple to search for all ToDos. 
        """
    )
    return


@app.cell
async def __(client):
    items = await client.store.search_items(
        ("todo", "general", "Test"),
        limit=5,
        offset=0
    )
    items['items']
    return (items,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Add items

        In our graph, we call `put` to add items to the store.

        We can use [put](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.put_item) with the SDK if we want to directly add items to the store outside our graph.
        """
    )
    return


@app.cell
async def __(client):
    from uuid import uuid4
    await client.store.put_item(
        ("testing", "Test"),
        key=str(uuid4()),
        value={"todo": "Test SDK put_item"},
    )
    return (uuid4,)


@app.cell
async def __(client):
    items_1 = await client.store.search_items(('testing', 'Test'), limit=5, offset=0)
    items_1['items']
    return (items_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Delete items

        We can use the SDK to [delete items](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.StoreClient.delete_item) from the store by key.
        """
    )
    return


@app.cell
def __(items_1):
    [item['key'] for item in items_1['items']]
    return


@app.cell
async def __(client):
    await client.store.delete_item(
           ("testing", "Test"),
            key='3de441ba-8c79-4beb-8f52-00e4dcba68d4',
        )
    return


@app.cell
async def __(client):
    items_2 = await client.store.search_items(('testing', 'Test'), limit=5, offset=0)
    items_2['items']
    return (items_2,)


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

