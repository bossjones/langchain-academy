import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/parallelization.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239934-lesson-1-parallelization)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Parallel node execution

        ## Review

        In module 3, we went in-depth on `human-in-the loop`, showing 3 common use-cases:

        (1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action

        (2) `Debugging` - We can rewind the graph to reproduce or avoid issues

        (3) `Editing` - You can modify the state 

        ## Goals

        This module will build on `human-in-the-loop` as well as the `memory` concepts discussed in module 2.

        We will dive into `multi-agent` workflows, and build up to a multi-agent research assistant that ties together all of the modules from this course.

        To build this multi-agent research assistant, we'll first discuss a few LangGraph controllability topics.

        We'll start with [parallelization](https://langchain-ai.github.io/langgraph/how-tos/branching/#how-to-create-branches-for-parallel-node-execution).

        ## Fan out and fan in

        Let's build a simple linear graph that over-writes the state at each step.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install -U  langgraph tavily-python wikipedia langchain_openai langchain_community langgraph_sdk
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
def __(builder):
    from IPython.display import Image, display
    from typing import Any
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        state: str

    class ReturnNodeValue:

        def __init__(self, node_secret: str):
            self._value = node_secret

        def __call__(self, state: State) -> Any:
            print(f'Adding {self._value} to {state['state']}')
            return {'state': [self._value]}
    _builder = StateGraph(State)
    _builder.add_node('a', ReturnNodeValue("I'm A"))
    _builder.add_node('b', ReturnNodeValue("I'm B"))
    _builder.add_node('c', ReturnNodeValue("I'm C"))
    _builder.add_node('d', ReturnNodeValue("I'm D"))
    _builder.add_edge(START, 'a')
    _builder.add_edge('a', 'b')
    _builder.add_edge('b', 'c')
    _builder.add_edge('c', 'd')
    _builder.add_edge('d', END)
    graph = builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        Any,
        END,
        Image,
        ReturnNodeValue,
        START,
        State,
        StateGraph,
        TypedDict,
        display,
        graph,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        We over-write state, as expected.
        """
    )
    return


@app.cell
def __(graph):
    graph.invoke({"state": []})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, let's run `b` and `c` in parallel. 

        And then run `d`.

        We can do this easily with fan-out from `a` to `b` and `c`, and then fan-in to `d`.

        The the state updates are applied at the end of each step.

        Let's run it.
        """
    )
    return


@app.cell
def __(
    END,
    Image,
    ReturnNodeValue,
    START,
    State,
    StateGraph,
    builder,
    display,
):
    _builder = StateGraph(State)
    _builder.add_node('a', ReturnNodeValue("I'm A"))
    _builder.add_node('b', ReturnNodeValue("I'm B"))
    _builder.add_node('c', ReturnNodeValue("I'm C"))
    _builder.add_node('d', ReturnNodeValue("I'm D"))
    _builder.add_edge(START, 'a')
    _builder.add_edge('a', 'b')
    _builder.add_edge('a', 'c')
    _builder.add_edge('b', 'd')
    _builder.add_edge('c', 'd')
    _builder.add_edge('d', END)
    graph_1 = builder.compile()
    display(Image(graph_1.get_graph().draw_mermaid_png()))
    return (graph_1,)


@app.cell
def __(mo):
    mo.md(
        r"""
        **We see an error**! 

        This is because both `b` and `c` are writing to the same state key / channel in the same step. 
        """
    )
    return


@app.cell
def __(graph_1):
    from langgraph.errors import InvalidUpdateError
    try:
        graph_1.invoke({'state': []})
    except InvalidUpdateError as e:
        print(f'An error occurred: {e}')
    return (InvalidUpdateError,)


@app.cell
def __(mo):
    mo.md(
        r"""
        When using fan out, we need to be sure that we are using a reducer if steps are writing to the same the channel / key. 

        As we touched on in Module 2, `operator.add` is a function from Python's built-in operator module.

        When `operator.add` is applied to lists, it performs list concatenation.
        """
    )
    return


@app.cell
def __(
    END,
    Image,
    ReturnNodeValue,
    START,
    StateGraph,
    TypedDict,
    builder,
    display,
):
    import operator
    from typing import Annotated

    class State_1(TypedDict):
        state: Annotated[list, operator.add]
    _builder = StateGraph(State_1)
    _builder.add_node('a', ReturnNodeValue("I'm A"))
    _builder.add_node('b', ReturnNodeValue("I'm B"))
    _builder.add_node('c', ReturnNodeValue("I'm C"))
    _builder.add_node('d', ReturnNodeValue("I'm D"))
    _builder.add_edge(START, 'a')
    _builder.add_edge('a', 'b')
    _builder.add_edge('a', 'c')
    _builder.add_edge('b', 'd')
    _builder.add_edge('c', 'd')
    _builder.add_edge('d', END)
    graph_2 = builder.compile()
    display(Image(graph_2.get_graph().draw_mermaid_png()))
    return Annotated, State_1, graph_2, operator


@app.cell
def __(graph_2):
    graph_2.invoke({'state': []})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now we see that we append to state for the updates made in parallel by `b` and `c`.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Waiting for nodes to finish

        Now, lets consider a case where one parallel path has more steps than the other one.
        """
    )
    return


@app.cell
def __(
    END,
    Image,
    ReturnNodeValue,
    START,
    StateGraph,
    State_1,
    builder,
    display,
):
    _builder = StateGraph(State_1)
    _builder.add_node('a', ReturnNodeValue("I'm A"))
    _builder.add_node('b', ReturnNodeValue("I'm B"))
    _builder.add_node('b2', ReturnNodeValue("I'm B2"))
    _builder.add_node('c', ReturnNodeValue("I'm C"))
    _builder.add_node('d', ReturnNodeValue("I'm D"))
    _builder.add_edge(START, 'a')
    _builder.add_edge('a', 'b')
    _builder.add_edge('a', 'c')
    _builder.add_edge('b', 'b2')
    _builder.add_edge(['b2', 'c'], 'd')
    _builder.add_edge('d', END)
    graph_3 = builder.compile()
    display(Image(graph_3.get_graph().draw_mermaid_png()))
    return (graph_3,)


@app.cell
def __(mo):
    mo.md(
        r"""
        In this case, `b`, `b2`, and `c` are all part of the same step.

        The graph will wait for all of these to be completed before proceeding to step `d`. 
        """
    )
    return


@app.cell
def __(graph_3):
    graph_3.invoke({'state': []})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Setting the order of state updates

        However, within each step we don't have specific control over the order of the state updates!

        In simple terms, it is a deterministic order determined by LangGraph based upon graph topology that **we do not control**. 

        Above, we see that `c` is added before `b2`.

        However, we can use a custom reducer to customize this e.g., sort state updates.
        """
    )
    return


@app.cell
def __(
    Annotated,
    END,
    Image,
    ReturnNodeValue,
    START,
    StateGraph,
    TypedDict,
    builder,
    display,
):
    def sorting_reducer(left, right):
        """ Combines and sorts the values in a list"""
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]
        return sorted(left + right, reverse=False)

    class State_2(TypedDict):
        state: Annotated[list, sorting_reducer]
    _builder = StateGraph(State_2)
    _builder.add_node('a', ReturnNodeValue("I'm A"))
    _builder.add_node('b', ReturnNodeValue("I'm B"))
    _builder.add_node('b2', ReturnNodeValue("I'm B2"))
    _builder.add_node('c', ReturnNodeValue("I'm C"))
    _builder.add_node('d', ReturnNodeValue("I'm D"))
    _builder.add_edge(START, 'a')
    _builder.add_edge('a', 'b')
    _builder.add_edge('a', 'c')
    _builder.add_edge('b', 'b2')
    _builder.add_edge(['b2', 'c'], 'd')
    _builder.add_edge('d', END)
    graph_4 = builder.compile()
    display(Image(graph_4.get_graph().draw_mermaid_png()))
    return State_2, graph_4, sorting_reducer


@app.cell
def __(graph_4):
    graph_4.invoke({'state': []})
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, the reducer sorts the updated state values!
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Working with LLMs

        Now, lets add a realistic example! 

        We want to gather context from two external sources (Wikipedia and Web-Seach) and have an LLM answer a question.
        """
    )
    return


@app.cell
def __():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return ChatOpenAI, llm


@app.cell
def __(Annotated, TypedDict, operator):
    class State_3(TypedDict):
        question: str
        answer: str
        context: Annotated[list, operator.add]
    return (State_3,)


@app.cell
def __(mo):
    mo.md(
        r"""
        You can try different web search tools. [Tavily](https://tavily.com/) is one nice option to consider, but ensure your `TAVILY_API_KEY` is set.
        """
    )
    return


@app.cell
def __(getpass, os):
    def _set_env(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"{var}: ")
    _set_env("TAVILY_API_KEY")
    return


@app.cell
def __(END, Image, START, StateGraph, State_3, builder, display, llm):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_community.document_loaders import WikipediaLoader
    from langchain_community.tools.tavily_search import TavilySearchResults

    def search_web(state):
        """ Retrieve docs from web search """
        tavily_search = TavilySearchResults(max_results=3)
        search_docs = tavily_search.invoke(state['question'])
        formatted_search_docs = '\n\n---\n\n'.join([f'<Document href="{doc['url']}"/>\n{doc['content']}\n</Document>' for doc in search_docs])
        return {'context': [formatted_search_docs]}

    def search_wikipedia(state):
        """ Retrieve docs from wikipedia """
        search_docs = WikipediaLoader(query=state['question'], load_max_docs=2).load()
        formatted_search_docs = '\n\n---\n\n'.join([f'<Document source="{doc.metadata['source']}" page="{doc.metadata.get('page', '')}"/>\n{doc.page_content}\n</Document>' for doc in search_docs])
        return {'context': [formatted_search_docs]}

    def generate_answer(state):
        """ Node to answer a question """
        context = state['context']
        question = state['question']
        answer_template = 'Answer the question {question} using this context: {context}'
        answer_instructions = answer_template.format(question=question, context=context)
        answer = llm.invoke([SystemMessage(content=answer_instructions)] + [HumanMessage(content=f'Answer the question.')])
        return {'answer': answer}
    _builder = StateGraph(State_3)
    _builder.add_node('search_web', search_web)
    _builder.add_node('search_wikipedia', search_wikipedia)
    _builder.add_node('generate_answer', generate_answer)
    _builder.add_edge(START, 'search_wikipedia')
    _builder.add_edge(START, 'search_web')
    _builder.add_edge('search_wikipedia', 'generate_answer')
    _builder.add_edge('search_web', 'generate_answer')
    _builder.add_edge('generate_answer', END)
    graph_5 = builder.compile()
    display(Image(graph_5.get_graph().draw_mermaid_png()))
    return (
        HumanMessage,
        SystemMessage,
        TavilySearchResults,
        WikipediaLoader,
        generate_answer,
        graph_5,
        search_web,
        search_wikipedia,
    )


@app.cell
def __(graph_5):
    result = graph_5.invoke({'question': "How were Nvidia's Q2 2024 earnings"})
    result['answer'].content
    return (result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Using with LangGraph API

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        Let's load our the above graph in the Studio UI, which uses `module-4/studio/parallelization.py` set in `module-4/studio/langgraph.json`.

        ![Screenshot 2024-08-29 at 3.05.13 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb10f43c3d4df239e0278_parallelization-1.png)

        Let's get the URL for the local deployment from Studio.
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
    client = get_client(url="http://localhost:63082")
    return client, get_client


@app.cell
async def __(client):
    thread = await client.threads.create()
    input_question = {"question": "How were Nvidia Q2 2024 earnings?"}
    async for event in client.runs.stream(thread["thread_id"], 
                                          assistant_id="parallelization", 
                                          input=input_question, 
                                          stream_mode="values"):
        # Check if answer has been added to state  
        answer = event.data.get('answer', None)
        if answer:
            print(answer['content'])
    return answer, event, input_question, thread


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

