import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/chatbot-summarization.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239436-lesson-5-chatbot-w-summarizing-messages-and-memory)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Chatbot with message summarization

        ## Review

        We've covered how to customize graph state schema and reducer. 
         
        We've also shown a number of ways to trim or filter messages in graph state. 

        ## Goals

        Now, let's take it one step further! 

        Rather than just trimming or filtering messages, we'll show how to use LLMs to produce a running summary of the conversation.
         
        This allows us to retain a compressed representation of the full conversation, rather than just removing it with trimming or filtering.

        We'll incorporate this summarization into a simple Chatbot.  

        And we'll equip that Chatbot with memory, supporting long-running conversations without incurring high token cost / latency. 
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install --quiet -U langchain_core langgraph langchain_openai
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
def __():
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o",temperature=0)
    return ChatOpenAI, model


@app.cell
def __(mo):
    mo.md(
        r"""
        We'll use `MessagesState`, as before.

        In addition to the built-in `messages` key, we'll now include a custom key (`summary`).
        """
    )
    return


@app.cell
def __():
    from langgraph.graph import MessagesState
    class State(MessagesState):
        summary: str
    return MessagesState, State


@app.cell
def __(mo):
    mo.md(
        r"""
        We'll define a node to call our LLM that incorporates a summary, if it exists, into the prompt.
        """
    )
    return


@app.cell
def __(State, model):
    from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

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
    return HumanMessage, RemoveMessage, SystemMessage, call_model


@app.cell
def __(mo):
    mo.md(
        r"""
        We'll define a node to produce a summary.

        Note, here we'll use `RemoveMessage` to filter our state after we've produced the summary.
        """
    )
    return


@app.cell
def __(HumanMessage, RemoveMessage, State, model):
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
    return (summarize_conversation,)


@app.cell
def __(mo):
    mo.md(
        r"""
        We'll add a conditional edge to determine whether to produce a summary based on the conversation length.
        """
    )
    return


@app.cell
def __(State):
    from langgraph.graph import END
    # Determine whether to end or summarize the conversation
    def should_continue(state: State):
        
        """Return the next node to execute."""
        
        messages = state["messages"]
        
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"
        
        # Otherwise we can just end
        return END
    return END, should_continue


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Adding memory

        Recall that [state is transient](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) to a single graph execution.

        This limits our ability to have multi-turn conversations with interruptions. 

        As introduced at the end of Module 1, we can use [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/) to address this! 
         
        LangGraph can use a checkpointer to automatically save the graph state after each step.

        This built-in persistence layer gives us memory, allowing LangGraph to pick up from the last state update. 

        As we previously showed, one of the easiest to work with is `MemorySaver`, an in-memory key-value store for Graph state.

        All we need to do is compile the graph with a checkpointer, and our graph has memory!
        """
    )
    return


@app.cell
def __(END, State, call_model, should_continue, summarize_conversation):
    from IPython.display import Image, display
    from langgraph.checkpoint.memory import MemorySaver
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
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        Image,
        MemorySaver,
        START,
        StateGraph,
        display,
        graph,
        memory,
        workflow,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Threads

        The checkpointer saves the state at each step as a checkpoint.

        These saved checkpoints can be grouped into a `thread` of conversation.

        Think about Slack as an analog: different channels carry different conversations.

        Threads are like Slack channels, capturing grouped collections of state (e.g., conversation).

        Below, we use `configurable` to set a thread ID.

        ![state.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbadf3b379c2ee621adfd1_chatbot-summarization1.png)
        """
    )
    return


@app.cell
def __(HumanMessage, graph, input_message):
    config = {'configurable': {'thread_id': '1'}}
    _input_message = HumanMessage(content="hi! I'm Lance")
    _output = graph.invoke({'messages': [input_message]}, config)
    for _m in _output['messages'][-1:]:
        _m.pretty_print()
    _input_message = HumanMessage(content="what's my name?")
    _output = graph.invoke({'messages': [input_message]}, config)
    for _m in _output['messages'][-1:]:
        _m.pretty_print()
    _input_message = HumanMessage(content='i like the 49ers!')
    _output = graph.invoke({'messages': [input_message]}, config)
    for _m in _output['messages'][-1:]:
        _m.pretty_print()
    return (config,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we don't yet have a summary of the state because we still have < = 6 messages.

        This was set in `should_continue`. 

        ```
            # If there are more than six messages, then we summarize the conversation
            if len(messages) > 6:
                return "summarize_conversation"
        ```

        We can pick up the conversation because we have the thread.
        """
    )
    return


@app.cell
def __(config, graph):
    graph.get_state(config).values.get("summary","")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The `config` with thread ID allows us to proceed from the previously logged state!
        """
    )
    return


@app.cell
def __(HumanMessage, config, graph, input_message):
    _input_message = HumanMessage(content="i like Nick Bosa, isn't he the highest paid defensive player?")
    _output = graph.invoke({'messages': [input_message]}, config)
    for _m in _output['messages'][-1:]:
        _m.pretty_print()
    return


@app.cell
def __(config, graph):
    graph.get_state(config).values.get("summary","")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## LangSmith

        Let's review the trace!
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

