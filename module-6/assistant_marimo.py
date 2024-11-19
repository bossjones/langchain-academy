import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Assistants 

        [Assistants](https://langchain-ai.github.io/langgraph/concepts/assistants/#resources) give developers a quick and easy way to modify and version agents for experimentation.

        ## Supplying configuration to the graph

        Our `task_maistro` graph is already set up to use assistants!

        It has a `configuration.py` file defined and loaded in the graph.

        We access configurable fields (`user_id`, `todo_category`, `task_maistro_role`) inside the graph nodes.

        ## Creating assistants 

        Now, what is a practical use-case for assistants with the `task_maistro` app that we've been building?

        For me, it's the ability to have separate ToDo lists for different categories of tasks. 

        For example, I want one assistant for my personal tasks and another for my work tasks.

        These are easily configurable using the `todo_category` and `task_maistro_role` configurable fields.

        ![Screenshot 2024-11-18 at 9.35.55 AM.png](attachment:b3fcbc65-0aba-480c-a0ab-fec3f28c75f4.png)
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
def __():
    from __future__ import annotations
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"



    from dotenv import load_dotenv


    # Load environment variables from .env file for API access
    load_dotenv(dotenv_path=".env", override=True)

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    return annotations, load_dotenv, os


@app.cell
def __(mo):
    mo.md(
        r"""
        This is the default assistant that we created when we deployed the graph.
        """
    )
    return


@app.cell
def __():
    from langgraph_sdk import get_client
    url_for_cli_deployment = "http://localhost:8123"
    client = get_client(url=url_for_cli_deployment)
    return client, get_client, url_for_cli_deployment


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Personal assistant

        This is the personal assistant that I'll use to manage my personal tasks.
        """
    )
    return


@app.cell
async def __(client):
    personal_assistant = await client.assistants.create(
        # "task_maistro" is the name of a graph we deployed
        "task_maistro",
        config={"configurable": {"todo_category": "personal"}}
    )
    print(personal_assistant)
    return (personal_assistant,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's update this assistant to include my `user_id` for convenience, [creating a new version of it](https://langchain-ai.github.io/langgraph/cloud/how-tos/assistant_versioning/#create-a-new-version-for-your-assistant). 
        """
    )
    return


@app.cell
async def __(
    client,
    configurations,
    personal_assistant,
    task_maistro_role,
):
    _task_maistro_role = 'You are a friendly and organized personal task assistant. Your main focus is helping users stay on top of their personal tasks and commitments. Specifically:\n\n- Help track and organize personal tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- Proactively ask for deadlines when new tasks are added without them\n- Maintain a supportive tone while helping the user stay accountable\n- Help prioritize tasks based on deadlines and importance\n\nYour communication style should be encouraging and helpful, never judgmental.\n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Would you like to add one to help us track it better?'
    _configurations = {'todo_category': 'personal', 'user_id': 'lance', 'task_maistro_role': task_maistro_role}
    personal_assistant_1 = await client.assistants.update(personal_assistant['assistant_id'], config={'configurable': configurations})
    print(personal_assistant_1)
    return (personal_assistant_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Work assistant

        Now, let's create a work assistant. I'll use this for my work tasks.
        """
    )
    return


@app.cell
async def __(client, configurations, task_maistro_role):
    _task_maistro_role = 'You are a focused and efficient work task assistant.\n\nYour main focus is helping users manage their work commitments with realistic timeframes.\n\nSpecifically:\n\n- Help track and organize work tasks\n- When providing a \'todo summary\':\n  1. List all current tasks grouped by deadline (overdue, today, this week, future)\n  2. Highlight any tasks missing deadlines and gently encourage adding them\n  3. Note any tasks that seem important but lack time estimates\n- When discussing new tasks, suggest that the user provide realistic time-frames based on task type:\n  • Developer Relations features: typically 1 day\n  • Course lesson reviews/feedback: typically 2 days\n  • Documentation sprints: typically 3 days\n- Help prioritize tasks based on deadlines and team dependencies\n- Maintain a professional tone while helping the user stay accountable\n\nYour communication style should be supportive but practical.\n\nWhen tasks are missing deadlines, respond with something like "I notice [task] doesn\'t have a deadline yet. Based on similar tasks, this might take [suggested timeframe]. Would you like to set a deadline with this in mind?'
    _configurations = {'todo_category': 'work', 'user_id': 'lance', 'task_maistro_role': task_maistro_role}
    work_assistant = await client.assistants.create('task_maistro', config={'configurable': configurations})
    print(work_assistant)
    return (work_assistant,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Using assistants 

        Assistants will be saved to `Postgres` in our deployment.  

        This allows us to easily search [search](https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/) for assistants with the SDK.
        """
    )
    return


@app.cell
async def __(client):
    assistants = await client.assistants.search()
    for assistant in assistants:
        print({
            'assistant_id': assistant['assistant_id'],
            'version': assistant['version'],
            'config': assistant['config']
        })
    return assistant, assistants


@app.cell
def __(mo):
    mo.md(
        r"""
        We can manage them easily with the SDK. For example, we can delete assistants that we're no longer using.
        """
    )
    return


@app.cell
async def __(client):
    await client.assistants.delete("assistant_id")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's set the assistant IDs for the `personal` and `work` assistants that I'll work with.
        """
    )
    return


@app.cell
def __(assistants):
    work_assistant_id = assistants[0]['assistant_id']
    personal_assistant_id = assistants[1]['assistant_id']
    return personal_assistant_id, work_assistant_id


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Work assistant

        Let's add some ToDos for my work assistant.
        """
    )
    return


@app.cell
async def __(chunk, client, work_assistant_id):
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import convert_to_messages
    _user_input = 'Create or update few ToDos: 1) Re-film Module 6, lesson 5 by end of day today. 2) Update audioUX by next Monday.'
    thread = await client.threads.create()
    async for _chunk in client.runs.stream(thread['thread_id'], work_assistant_id, input={'messages': [HumanMessage(content=_user_input)]}, stream_mode='values'):
        if _chunk.event == 'values':
            _state = chunk.data
            convert_to_messages(_state['messages'])[-1].pretty_print()
    return HumanMessage, convert_to_messages, thread


@app.cell
async def __(
    HumanMessage,
    chunk,
    client,
    convert_to_messages,
    work_assistant_id,
):
    _user_input = 'Create another ToDo: Finalize set of report generation tutorials.'
    thread_1 = await client.threads.create()
    async for _chunk in client.runs.stream(thread_1['thread_id'], work_assistant_id, input={'messages': [HumanMessage(content=_user_input)]}, stream_mode='values'):
        if _chunk.event == 'values':
            _state = chunk.data
            convert_to_messages(_state['messages'])[-1].pretty_print()
    return (thread_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        The assistant uses it's instructions to push back with task creation! 

        It asks me to specify a deadline :) 
        """
    )
    return


@app.cell
async def __(
    HumanMessage,
    chunk,
    client,
    convert_to_messages,
    thread_1,
    work_assistant_id,
):
    _user_input = "OK, for this task let's get it done by next Tuesday."
    async for _chunk in client.runs.stream(thread_1['thread_id'], work_assistant_id, input={'messages': [HumanMessage(content=_user_input)]}, stream_mode='values'):
        if _chunk.event == 'values':
            _state = chunk.data
            convert_to_messages(_state['messages'])[-1].pretty_print()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Personal assistant

        Similarly, we can add ToDos for my personal assistant.
        """
    )
    return


@app.cell
async def __(
    HumanMessage,
    chunk,
    client,
    convert_to_messages,
    personal_assistant_id,
):
    _user_input = 'Create ToDos: 1) Check on swim lessons for the baby this weekend. 2) For winter travel, check AmEx points.'
    thread_2 = await client.threads.create()
    async for _chunk in client.runs.stream(thread_2['thread_id'], personal_assistant_id, input={'messages': [HumanMessage(content=_user_input)]}, stream_mode='values'):
        if _chunk.event == 'values':
            _state = chunk.data
            convert_to_messages(_state['messages'])[-1].pretty_print()
    return (thread_2,)


@app.cell
async def __(
    HumanMessage,
    chunk,
    client,
    convert_to_messages,
    personal_assistant_id,
):
    _user_input = 'Give me a todo summary.'
    thread_3 = await client.threads.create()
    async for _chunk in client.runs.stream(thread_3['thread_id'], personal_assistant_id, input={'messages': [HumanMessage(content=_user_input)]}, stream_mode='values'):
        if _chunk.event == 'values':
            _state = chunk.data
            convert_to_messages(_state['messages'])[-1].pretty_print()
    return (thread_3,)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

