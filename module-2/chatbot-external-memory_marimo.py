import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/chatbot-external-memory.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239440-lesson-6-chatbot-w-summarizing-messages-and-external-memory)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Chatbot with message summarization & external DB memory

        ## Review

        We've covered how to customize graph state schema and reducer. 
         
        We've also shown a number of tricks for trimming or filtering messages in graph state. 

        We've used these concepts in a Chatbot with memory that produces a running summary of the conversation.

        ## Goals

        But, what if we want our Chatbot to have memory that persists indefinitely?

        Now, we'll introduce some more advanced checkpointers that support external databases. 

        Here, we'll show how to use [Sqlite as a checkpointer](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer), but other checkpointers, such as [Postgres](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/) are available!
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install --quiet -U langgraph-checkpoint-sqlite langchain_core langgraph langchain_openai
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
        ## Sqlite

        A good starting point here is the [SqliteSaver checkpointer](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer).

        Sqlite is a [small, fast, highly popular](https://x.com/karpathy/status/1819490455664685297) SQL database. 
         
        If we supply `":memory:"` it creates an in-memory Sqlite database.
        """
    )
    return


@app.cell
def __():
    import sqlite3
    # In memory
    conn = sqlite3.connect(":memory:", check_same_thread = False)
    return conn, sqlite3


@app.cell
def __(mo):
    mo.md(
        r"""
        But, if we supply a db path, then it will create a database for us!
        """
    )
    return


app._unparsable_cell(
    r"""
    # pull file if it doesn't exist and connect to local db
    !mkdir -p state_db && [ ! -f state_db/example.db ] && wget -P state_db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db

    db_path = \"state_db/example.db\"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    """,
    name="__"
)


@app.cell
def __(conn):
    # Here is our checkpointer 
    from langgraph.checkpoint.sqlite import SqliteSaver
    memory = SqliteSaver(conn)
    return SqliteSaver, memory


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's re-define our chatbot.
        """
    )
    return


@app.cell
def __():
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

    from langgraph.graph import END
    from langgraph.graph import MessagesState

    model = ChatOpenAI(model="gpt-4o",temperature=0)

    class State(MessagesState):
        summary: str

    # Define the logic to call the model
    def call_model(state: State):
        
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
        
        response = model.invoke(messages)
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
    return (
        ChatOpenAI,
        END,
        HumanMessage,
        MessagesState,
        RemoveMessage,
        State,
        SystemMessage,
        call_model,
        model,
        should_continue,
        summarize_conversation,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we just re-compile with our sqlite checkpointer.
        """
    )
    return


@app.cell
def __(
    END,
    State,
    call_model,
    memory,
    should_continue,
    summarize_conversation,
):
    from IPython.display import Image, display
    from langgraph.graph import StateGraph, START

    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    graph = workflow.compile(checkpointer=memory)
    display(Image(graph.get_graph().draw_mermaid_png()))
    return Image, START, StateGraph, display, graph, workflow


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we can invoke the graph several times. 
        """
    )
    return


@app.cell
def __(HumanMessage, graph):
    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    # Start conversation
    input_message = HumanMessage(content="hi! I'm Lance")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="what's my name?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="i like the 49ers!")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-1:]:
        m.pretty_print()
    return config, input_message, m, output


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's confirm that our state is saved locally.
        """
    )
    return


@app.cell
def __(graph):
    config_1 = {'configurable': {'thread_id': '1'}}
    graph_state = graph.get_state(config_1)
    graph_state
    return config_1, graph_state


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Persisting state

        Using database like Sqlite means state is persisted! 

        For example, we can re-start the notebook kernel and see that we can still load from Sqlite DB on disk.

        """
    )
    return


@app.cell
def __(graph):
    config_2 = {'configurable': {'thread_id': '1'}}
    graph_state_1 = graph.get_state(config_2)
    graph_state_1
    return config_2, graph_state_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ## LangGraph Studio

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        --

        Now that we better understand external memory, recall that the LangGraph API packages your code and provides you with with built-in persistence.
         
        And the API is the back-end for Studio! 

        Load the `chatbot` in the UI, which uses `module2-/studio/chatbot.py` set in `module2-/studio/langgraph.json`.
        """
    )
    return


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

