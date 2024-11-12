import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239303-lesson-8-deployment)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Deployment

        ## Review 

        We built up to an agent with memory: 

        * `act` - let the model call specific tools 
        * `observe` - pass the tool output back to the model 
        * `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)
        * `persist state` - use an in memory checkpointer to support long-running conversations with interruptions

        ## Goals

        Now, we'll cover how to actually deploy our agent locally to Studio and to `LangGraph Cloud`.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install --quiet -U langgraph_sdk langchain_core
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Concepts

        There are a few central concepts to understand -

        `LangGraph` —
        - Python and JavaScript library 
        - Allows creation of agent workflows 

        `LangGraph API` —
        - Bundles the graph code 
        - Provides a task queue for managing asynchronous operations
        - Offers persistence for maintaining state across interactions

        `LangGraph Cloud` --
        - Hosted service for the LangGraph API
        - Allows deployment of graphs from GitHub repositories
        - Also provides monitoring and tracing for deployed graphs
        - Accessible via a unique URL for each deployment

        `LangGraph Studio` --
        - Integrated Development Environment (IDE) for LangGraph applications
        - Uses the API as its back-end, allowing real-time testing and exploration of graphs
        - Can be run locally or with cloud-deployment

        `LangGraph SDK` --
        - Python library for programmatically interacting with LangGraph graphs
        - Provides a consistent interface for working with graphs, whether served locally or in the cloud
        - Allows creation of clients, access to assistants, thread management, and execution of runs

        ## Testing Locally

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        We can easily connect with graphs that are served locally in LangGraph Studio!

        We do this via the `url` provided in the lower left corner of the Studio UI.

        ![Screenshot 2024-08-23 at 1.17.05 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbad4f53080e6802cec34d_deployment%201.png)
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
def __():
    from langgraph_sdk import get_client
    return (get_client,)


@app.cell
async def __(URL, get_client):
    _URL = 'http://localhost:56091'
    client = get_client(url=URL)
    assistants = await client.assistants.search()
    return assistants, client


@app.cell
def __(assistants):
    assistants[-3]
    return


@app.cell
async def __(client):
    # We create a thread for tracking the state of our run
    thread = await client.threads.create()
    return (thread,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can run our agent [with `client.runs.stream`](https://langchain-ai.github.io/langgraph/concepts/low_level/#stream-and-astream) with:

        * The `thread_id`
        * The `graph_id`
        * The `input` 
        * The `stream_mode`

        We'll discuss streaming in depth in a future module. 

        For now, just recognize that we are [streaming](https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_values/) the full value of the state after each step of the graph with `stream_mode="values"`.
         
        The state is captured in the `chunk.data`. 
        """
    )
    return


@app.cell
async def __(client, thread):
    from langchain_core.messages import HumanMessage
    _input = {'messages': [HumanMessage(content='Multiply 3 by 2.')]}
    async for _chunk in client.runs.stream(thread['thread_id'], 'agent', input=_input, stream_mode='values'):
        if _chunk.data and _chunk.event != 'metadata':
            print(_chunk.data['messages'][-1])
    return (HumanMessage,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Testing with Cloud

        We can deploy to Cloud via LangSmith, as outlined [here](https://langchain-ai.github.io/langgraph/cloud/quick_start/#deploy-from-github-with-langgraph-cloud). 

        ### Create a New Repository on GitHub

        * Go to your GitHub account
        * Click on the "+" icon in the upper-right corner and select `"New repository"`
        * Name your repository (e.g., `langchain-academy`)

        ### Add Your GitHub Repository as a Remote

        * Go back to your terminal where you cloned `langchain-academy` at the start of this course
        * Add your newly created GitHub repository as a remote

        ```
        git remote add origin https://github.com/your-username/your-repo-name.git
        ```
        * Push to it
        ```
        git push -u origin main
        ```

        ### Connect LangSmith to your GitHub Repository

        * Go to [LangSmith](hhttps://smith.langchain.com/)
        * Click on `deployments` tab on the left LangSmith panel
        * Add `+ New Deployment`
        * Then, select the Github repository (e.g., `langchain-academy`) that you just created for the course
        * Point the `LangGraph API config file` at one of the `studio` directories
        * For example, for module-1 use: `module-1/studio/langgraph.json`
        * Set your API keys (e.g., you can just copy from your `module-1/studio/.env` file)

        ![Screenshot 2024-09-03 at 11.35.12 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbad4fd61c93d48e5d0f47_deployment2.png)

        ### Work with your deployment

        We can then interact with our deployment a few different ways:

        * With the [SDK](https://langchain-ai.github.io/langgraph/cloud/quick_start/#use-with-the-sdk), as before.
        * With [LangGraph Studio](https://langchain-ai.github.io/langgraph/cloud/quick_start/#interact-with-your-deployment-via-langgraph-studio).

        ![Screenshot 2024-08-23 at 10.59.36 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbad4fa159a09a51d601de_deployment3.png)

        To use the SDK here in the notebook, simply ensure that `LANGSMITH_API_KEY` is set!
        """
    )
    return


@app.cell
def __():
    import os, getpass

    def _set_env(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"{var}: ")

    _set_env("LANGCHAIN_API_KEY")
    return getpass, os


@app.cell
async def __(URL, get_client):
    _URL = 'https://langchain-academy-8011c561878d50b1883f7ed11b32d720.default.us.langgraph.app'
    client_1 = get_client(url=URL)
    assistants_1 = await client_1.assistants.search()
    return assistants_1, client_1


@app.cell
def __(assistants_1):
    agent = assistants_1[0]
    return (agent,)


@app.cell
def __(agent):
    agent
    return


@app.cell
async def __(HumanMessage, client_1):
    thread_1 = await client_1.threads.create()
    _input = {'messages': [HumanMessage(content='Multiply 3 by 2.')]}
    async for _chunk in client_1.runs.stream(thread_1['thread_id'], 'agent', input=_input, stream_mode='values'):
        if _chunk.data and _chunk.event != 'metadata':
            print(_chunk.data['messages'][-1])
    return (thread_1,)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

