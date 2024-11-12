import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Sub-graphs

        ## Review

        We're building up to a multi-agent research assistant that ties together all of the modules from this course.

        We just covered parallelization, which is one important LangGraph controllability topic.

        ## Goals

        Now, we're [going to cover sub-graphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/#simple-example).

        ## State

        Sub-graphs allow you to create and manage different states in different parts of your graph. 

        This is particularly useful for multi-agent systems, with teams of agents that each have their own state.

        Let's consider a toy example:

        * I have a system that accepts logs
        * It performs two separate sub-tasks by different agents (summarize logs, find failure modes)
        * I want to perform these two operations in two different sub-graphs.

        The most critical thing to understand is how the graphs communicate! 

        In short, communication is **done with over-lapping keys**: 

        * The sub-graphs can access `docs` from the parent
        * The parent can access `summary/failure_report` from the sub-graphs

        ![subgraph.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb1abf89f2d847ee6f1ff_sub-graph1.png)

        ## Input

        Let's define a schema for the logs that will be input to our graph.
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install -U  langgraph
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        We'll use [LangSmith](https://docs.smith.langchain.com/) for [tracing](https://docs.smith.langchain.com/concepts/tracing).
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
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"
    return getpass, os


@app.cell
def __():
    from operator import add
    from typing_extensions import TypedDict
    from typing import List, Optional, Annotated

    # The structure of the logs
    class Log(TypedDict):
        id: str
        question: str
        docs: Optional[List]
        answer: str
        grade: Optional[int]
        grader: Optional[str]
        feedback: Optional[str]
    return Annotated, List, Log, Optional, TypedDict, add


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Sub graphs

        Here is the failure analysis sub-graph, which uses `FailureAnalysisState`.
        """
    )
    return


@app.cell
def __(List, Log, TypedDict):
    from IPython.display import Image, display
    from langgraph.graph import StateGraph, START, END

    class FailureAnalysisState(TypedDict):
        cleaned_logs: List[Log]
        failures: List[Log]
        fa_summary: str
        processed_logs: List[str]

    class FailureAnalysisOutputState(TypedDict):
        fa_summary: str
        processed_logs: List[str]

    def get_failures(state):
        """ Get logs that contain a failure """
        cleaned_logs = state['cleaned_logs']
        failures = [log for log in cleaned_logs if 'grade' in log]
        return {'failures': failures}

    def _generate_summary(state):
        """ Generate summary of failures """
        failures = state['failures']
        fa_summary = 'Poor quality retrieval of Chroma documentation.'
        return {'fa_summary': fa_summary, 'processed_logs': [f'failure-analysis-on-log-{failure['id']}' for failure in failures]}
    fa_builder = StateGraph(input=FailureAnalysisState, output=FailureAnalysisOutputState)
    fa_builder.add_node('get_failures', get_failures)
    fa_builder.add_node('generate_summary', _generate_summary)
    fa_builder.add_edge(START, 'get_failures')
    fa_builder.add_edge('get_failures', 'generate_summary')
    fa_builder.add_edge('generate_summary', END)
    graph = fa_builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return (
        END,
        FailureAnalysisOutputState,
        FailureAnalysisState,
        Image,
        START,
        StateGraph,
        display,
        fa_builder,
        get_failures,
        graph,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Here is the question summarization sub-grap, which uses `QuestionSummarizationState`.
        """
    )
    return


@app.cell
def __(END, Image, List, Log, START, StateGraph, TypedDict, display):
    class QuestionSummarizationState(TypedDict):
        cleaned_logs: List[Log]
        qs_summary: str
        report: str
        processed_logs: List[str]

    class QuestionSummarizationOutputState(TypedDict):
        report: str
        processed_logs: List[str]

    def _generate_summary(state):
        cleaned_logs = state['cleaned_logs']
        summary = 'Questions focused on usage of ChatOllama and Chroma vector store.'
        return {'qs_summary': summary, 'processed_logs': [f'summary-on-log-{log['id']}' for log in cleaned_logs]}

    def send_to_slack(state):
        qs_summary = state['qs_summary']
        report = 'foo bar baz'
        return {'report': report}
    qs_builder = StateGraph(input=QuestionSummarizationState, output=QuestionSummarizationOutputState)
    qs_builder.add_node('generate_summary', _generate_summary)
    qs_builder.add_node('send_to_slack', send_to_slack)
    qs_builder.add_edge(START, 'generate_summary')
    qs_builder.add_edge('generate_summary', 'send_to_slack')
    qs_builder.add_edge('send_to_slack', END)
    graph_1 = qs_builder.compile()
    display(Image(graph_1.get_graph().draw_mermaid_png()))
    return (
        QuestionSummarizationOutputState,
        QuestionSummarizationState,
        graph_1,
        qs_builder,
        send_to_slack,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Adding sub graphs to our parent graph

        Now, we can bring it all together.

        We create our parent graph with `EntryGraphState`. 

        And we add our sub-graphs as nodes! 

        ```
        entry_builder.add_node("question_summarization", qs_builder.compile())
        entry_builder.add_node("failure_analysis", fa_builder.compile())
        ```
        """
    )
    return


@app.cell
def __(Annotated, List, Log, TypedDict, add):
    class _EntryGraphState(TypedDict):
        raw_logs: List[Log]
        cleaned_logs: Annotated[List[Log], add]
        fa_summary: str
        report: str
        processed_logs: Annotated[List[int], add]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        But, why does `cleaned_logs` have a reducer if it only goes *into* each sub-graph as an input? It is not modified.

        ```
        cleaned_logs: Annotated[List[Log], add] # This will be USED BY in BOTH sub-graphs
        ```

        This is because the output state of the subgraphs will contain **all keys**, even if they are unmodified. 

        The sub-graphs are run in parallel.

        Because the parallel sub-graphs return the same key, it needs to have a reducer like `operator.add` to combine the incoming values from each sub-graph.

        But, we can work around this by using another concept we talked about before.

        We can simply create an output state schema for each sub-graph and ensure that the output state schema contains different keys to publish as output.

        We don't actually need each sub-graph to output `cleaned_logs`.
        """
    )
    return


@app.cell
def __(
    Annotated,
    END,
    EntryGraphState,
    Image,
    List,
    Log,
    START,
    StateGraph,
    TypedDict,
    add,
    display,
    fa_builder,
    qs_builder,
):
    class _EntryGraphState(TypedDict):
        raw_logs: List[Log]
        cleaned_logs: List[Log]
        fa_summary: str
        report: str
        processed_logs: Annotated[List[int], add]

    def clean_logs(state):
        raw_logs = state['raw_logs']
        cleaned_logs = raw_logs
        return {'cleaned_logs': cleaned_logs}
    entry_builder = StateGraph(EntryGraphState)
    entry_builder.add_node('clean_logs', clean_logs)
    entry_builder.add_node('question_summarization', qs_builder.compile())
    entry_builder.add_node('failure_analysis', fa_builder.compile())
    entry_builder.add_edge(START, 'clean_logs')
    entry_builder.add_edge('clean_logs', 'failure_analysis')
    entry_builder.add_edge('clean_logs', 'question_summarization')
    entry_builder.add_edge('failure_analysis', END)
    entry_builder.add_edge('question_summarization', END)
    graph_2 = entry_builder.compile()
    display(Image(graph_2.get_graph(xray=1).draw_mermaid_png()))
    return clean_logs, entry_builder, graph_2


@app.cell
def __(Log, graph_2):
    question_answer = Log(id='1', question='How can I import ChatOllama?', answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'")
    question_answer_feedback = Log(id='2', question='How can I use Chroma vector store?', answer='To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).', grade=0, grader='Document Relevance Recall', feedback='The retrieved documents discuss vector stores in general, but not Chroma specifically')
    raw_logs = [question_answer, question_answer_feedback]
    graph_2.invoke({'raw_logs': raw_logs})
    return question_answer, question_answer_feedback, raw_logs


@app.cell
def __(mo):
    mo.md(
        r"""
        ## LangSmith

        Let's look at the LangSmith trace:

        https://smith.langchain.com/public/f8f86f61-1b30-48cf-b055-3734dfceadf2/r
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

