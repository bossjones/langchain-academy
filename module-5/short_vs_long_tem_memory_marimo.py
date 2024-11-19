import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Lession 1: Short vs Long-Term Memory
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # # %pip install -U langchain_openai langgraph trustcall langchain_core
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ![title](images/1.png)
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
def __(mo):
    mo.md(
        r"""
        ![title](images/2.png)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Long Term Memory: human memory

        `semantic` - facts what I learned in school

        `procedural` - how to do something, how to ride a bike

        `episodic` - how to ride a bike

        # Long Term Memory: Agents memory

        `semantic` - facts about a user

        `procedural` - past agent actions

        `episodic` - agent's system prompt
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

