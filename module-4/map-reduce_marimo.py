import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/map-reduce.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239947-lesson-3-map-reduce)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Map-reduce

        ## Review

        We're building up to a multi-agent research assistant that ties together all of the modules from this course.

        To build this multi-agent assistant, we've been introducing a few LangGraph controllability topics.

        We just covered parallelization and sub-graphs.

        ## Goals

        Now, we're going to cover [map reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/).
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install -U langchain_openai langgraph
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
def __(mo):
    mo.md(
        r"""
        ## Problem

        Map-reduce operations are essential for efficient task decomposition and parallel processing. 

        It has two phases:

        (1) `Map` - Break a task into smaller sub-tasks, processing each sub-task in parallel.

        (2) `Reduce` - Aggregate the results across all of the completed, parallelized sub-tasks.

        Let's design a system that will do two things:

        (1) `Map` - Create a set of jokes about a topic.

        (2) `Reduce` - Pick the best joke from the list.

        We'll use an LLM to do the job generation and selection.
        """
    )
    return


@app.cell
def __():
    from langchain_openai import ChatOpenAI

    # Prompts we will use
    subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
    joke_prompt = """Generate a joke about {subject}"""
    best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

    # LLM
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    return ChatOpenAI, best_joke_prompt, joke_prompt, model, subjects_prompt


@app.cell
def __(mo):
    mo.md(
        r"""
        ## State

        ### Parallelizing joke generation

        First, let's define the entry point of the graph that will:

        * Take a user input topic
        * Produce a list of joke topics from it
        * Send each joke topic to our above joke generation node

        Our state has a `jokes` key, which will accumulate jokes from parallelized joke generation
        """
    )
    return


@app.cell
def __():
    import operator
    from typing import Annotated
    from typing_extensions import TypedDict
    from pydantic import BaseModel

    class Subjects(BaseModel):
        subjects: list[str]

    class BestJoke(BaseModel):
        id: int
        
    class OverallState(TypedDict):
        topic: str
        subjects: list
        jokes: Annotated[list, operator.add]
        best_selected_joke: str
    return (
        Annotated,
        BaseModel,
        BestJoke,
        OverallState,
        Subjects,
        TypedDict,
        operator,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Generate subjects for jokes.
        """
    )
    return


@app.cell
def __(OverallState, Subjects, model, subjects_prompt):
    def generate_topics(state: OverallState):
        prompt = subjects_prompt.format(topic=state["topic"])
        response = model.with_structured_output(Subjects).invoke(prompt)
        return {"subjects": response.subjects}
    return (generate_topics,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Here is the magic: we use the [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) to create a joke for each subject.

        This is very useful! It can automatically parallelize joke generation for any number of subjects.

        * `generate_joke`: the name of the node in the graph
        * `{"subject": s`}: the state to send

        `Send` allow you to pass any state that you want to `generate_joke`! It does not have to align with `OverallState`.

        In this case, `generate_joke` is using its own internal state, and we can popular this via `Send`.
        """
    )
    return


@app.cell
def __(OverallState):
    from langgraph.constants import Send
    def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
    return Send, continue_to_jokes


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Joke generation (map)

        Now, we just define a node that will create our jokes, `generate_joke`!

        We write them back out to `jokes` in `OverallState`! 

        This key has a reducer that will combine lists.
        """
    )
    return


@app.cell
def __(BaseModel, TypedDict, joke_prompt, model):
    class JokeState(TypedDict):
        subject: str

    class Joke(BaseModel):
        joke: str

    def generate_joke(state: JokeState):
        prompt = joke_prompt.format(subject=state["subject"])
        response = model.with_structured_output(Joke).invoke(prompt)
        return {"jokes": [response.joke]}
    return Joke, JokeState, generate_joke


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Best joke selection (reduce)

        Now, we add logic to pick the best joke.
        """
    )
    return


@app.cell
def __(BestJoke, OverallState, best_joke_prompt, model):
    def best_joke(state: OverallState):
        jokes = "\n\n".join(state["jokes"])
        prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
        response = model.with_structured_output(BestJoke).invoke(prompt)
        return {"best_selected_joke": state["jokes"][response.id]}
    return (best_joke,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Compile
        """
    )
    return


@app.cell
def __(
    OverallState,
    best_joke,
    continue_to_jokes,
    generate_joke,
    generate_topics,
):
    from IPython.display import Image
    from langgraph.graph import END, StateGraph, START

    # Construct the graph: here we put everything together to construct our graph
    graph = StateGraph(OverallState)
    graph.add_node("generate_topics", generate_topics)
    graph.add_node("generate_joke", generate_joke)
    graph.add_node("best_joke", best_joke)
    graph.add_edge(START, "generate_topics")
    graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
    graph.add_edge("generate_joke", "best_joke")
    graph.add_edge("best_joke", END)

    # Compile the graph
    app = graph.compile()
    Image(app.get_graph().draw_mermaid_png())
    return END, Image, START, StateGraph, app, graph


@app.cell
def __(app):
    # Call the graph: here we call it to generate a list of jokes
    for s in app.stream({"topic": "animals"}):
        print(s)
    return (s,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Studio

        --

        **⚠️ DISCLAIMER**

        *Running Studio currently requires a Mac. If you are not using a Mac, then skip this step.*

        *Also, if you are running this notebook in CoLab, then skip this step.*

        --

        Let's load our the above graph in the Studio UI, which uses `module-4/studio/map_reduce.py` set in `module-4/studio/langgraph.json`.

        ![Screenshot 2024-08-28 at 3.17.53 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb0c0ed88a12e822811e2_map-reduce1.png)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

