import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Double Texting

        Seamless handling of [double texting](https://langchain-ai.github.io/langgraph/concepts/double_texting/) is important for handling real-world usage scenarios, especially in chat applications.

        Users can send multiple messages in a row before the prior run(s) complete, and we want to ensure that we handle this gracefully.

        ## Reject

        A simple approach is to [reject](https://langchain-ai.github.io/langgraph/cloud/how-tos/reject_concurrent/) any new runs until the current run completes.
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
    url_for_cli_deployment = "http://localhost:8123"
    client = get_client(url=url_for_cli_deployment)
    return client, get_client, url_for_cli_deployment


@app.cell
async def __(client, config, graph_name, user_input_1):
    import httpx
    from langchain_core.messages import HumanMessage
    thread = await client.threads.create()
    _user_input_1 = 'Add a ToDo to follow-up with DI Repairs.'
    _user_input_2 = 'Add a ToDo to mount dresser to the wall.'
    _config = {'configurable': {'user_id': 'Test-Double-Texting'}}
    _graph_name = 'task_maistro'
    run = await client.runs.create(thread['thread_id'], graph_name, input={'messages': [HumanMessage(content=user_input_1)]}, config=config)
    try:
        await client.runs.create(thread['thread_id'], _graph_name, input={'messages': [HumanMessage(content=_user_input_2)]}, config=_config, multitask_strategy='reject')
    except httpx.HTTPStatusError as e:
        print('Failed to start concurrent run', e)
    return HumanMessage, httpx, run, thread


@app.cell
async def __(client, run, thread):
    from langchain_core.messages import convert_to_messages
    await client.runs.join(thread['thread_id'], run['run_id'])
    _state = await client.threads.get_state(thread['thread_id'])
    for _m in convert_to_messages(_state['values']['messages']):
        _m.pretty_print()
    return (convert_to_messages,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Enqueue

        We can use [enqueue](https://langchain-ai.github.io/langgraph/cloud/how-tos/enqueue_concurrent/https://langchain-ai.github.io/langgraph/cloud/how-tos/enqueue_concurrent/) any new runs until the current run completes.
        """
    )
    return


@app.cell
async def __(
    HumanMessage,
    client,
    config,
    convert_to_messages,
    graph_name,
    user_input_1,
    user_input_2,
):
    thread_1 = await client.threads.create()
    _user_input_1 = 'Send Erik his t-shirt gift this weekend.'
    _user_input_2 = 'Get cash and pay nanny for 2 weeks. Do this by Friday.'
    _config = {'configurable': {'user_id': 'Test-Double-Texting'}}
    _graph_name = 'task_maistro'
    first_run = await client.runs.create(thread_1['thread_id'], graph_name, input={'messages': [HumanMessage(content=user_input_1)]}, config=config)
    _second_run = await client.runs.create(thread_1['thread_id'], graph_name, input={'messages': [HumanMessage(content=user_input_2)]}, config=config, multitask_strategy='enqueue')
    await client.runs.join(thread_1['thread_id'], _second_run['run_id'])
    _state = await client.threads.get_state(thread_1['thread_id'])
    for _m in convert_to_messages(_state['values']['messages']):
        _m.pretty_print()
    return first_run, thread_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Interrupt

        We can use [interrupt](https://langchain-ai.github.io/langgraph/cloud/how-tos/interrupt_concurrent/) to interrupt the current run, but save all the work that has been done so far up to that point.

        """
    )
    return


@app.cell
async def __(
    HumanMessage,
    client,
    config,
    convert_to_messages,
    graph_name,
    user_input_1,
    user_input_2,
):
    import asyncio
    thread_2 = await client.threads.create()
    _user_input_1 = 'Give me a summary of my ToDos due tomrrow.'
    _user_input_2 = 'Never mind, create a ToDo to Order Ham for Thanksgiving by next Friday.'
    _config = {'configurable': {'user_id': 'Test-Double-Texting'}}
    _graph_name = 'task_maistro'
    interrupted_run = await client.runs.create(thread_2['thread_id'], graph_name, input={'messages': [HumanMessage(content=user_input_1)]}, config=config)
    await asyncio.sleep(1)
    _second_run = await client.runs.create(thread_2['thread_id'], graph_name, input={'messages': [HumanMessage(content=user_input_2)]}, config=config, multitask_strategy='interrupt')
    await client.runs.join(thread_2['thread_id'], _second_run['run_id'])
    _state = await client.threads.get_state(thread_2['thread_id'])
    for _m in convert_to_messages(_state['values']['messages']):
        _m.pretty_print()
    return asyncio, interrupted_run, thread_2


@app.cell
def __(mo):
    mo.md(
        r"""
        We can see the initial run is saved, and has status `interrupted`.
        """
    )
    return


@app.cell
async def __(client, interrupted_run, thread_2):
    print((await client.runs.get(thread_2['thread_id'], interrupted_run['run_id']))['status'])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Rollback

        We can use [rollback](https://langchain-ai.github.io/langgraph/cloud/how-tos/rollback_concurrent/) to interrupt the prior run of the graph, delete it, and start a new run with the double-texted input.

        """
    )
    return


@app.cell
async def __(
    HumanMessage,
    client,
    config,
    convert_to_messages,
    graph_name,
    user_input_1,
    user_input_2,
):
    thread_3 = await client.threads.create()
    _user_input_1 = 'Add a ToDo to call to make appointment at Yoga.'
    _user_input_2 = 'Actually, add a ToDo to drop by Yoga in person on Sunday.'
    _config = {'configurable': {'user_id': 'Test-Double-Texting'}}
    _graph_name = 'task_maistro'
    rolled_back_run = await client.runs.create(thread_3['thread_id'], graph_name, input={'messages': [HumanMessage(content=user_input_1)]}, config=config)
    _second_run = await client.runs.create(thread_3['thread_id'], graph_name, input={'messages': [HumanMessage(content=user_input_2)]}, config=config, multitask_strategy='rollback')
    await client.runs.join(thread_3['thread_id'], _second_run['run_id'])
    _state = await client.threads.get_state(thread_3['thread_id'])
    for _m in convert_to_messages(_state['values']['messages']):
        _m.pretty_print()
    return rolled_back_run, thread_3


@app.cell
def __(mo):
    mo.md(
        r"""
        The initial run was deleted.
        """
    )
    return


@app.cell
async def __(client, httpx, rolled_back_run, thread_3):
    try:
        await client.runs.get(thread_3['thread_id'], rolled_back_run['run_id'])
    except httpx.HTTPStatusError as _:
        print('Original run was correctly deleted')
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Summary 

        We can see [all the methods summarized](https://langchain-ai.github.io/langgraph/concepts/double_texting/):

        ![Screenshot 2024-11-15 at 12.13.18 PM.png](attachment:ff0af98b-71b1-497a-9c0e-b3519662fd2c.png)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

