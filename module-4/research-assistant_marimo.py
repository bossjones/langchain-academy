import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/research-assistant.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239974-lesson-4-research-assistant)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Research Assistant

        ## Review

        We've covered a few major LangGraph themes:

        * Memory
        * Human-in-the-loop
        * Controllability

        Now, we'll bring these ideas together to tackle one of AI's most popular applications: research automation. 

        Research is often laborious work offloaded to analysts. AI has considerable potential to assist with this.

        However, research demands customization: raw LLM outputs are often poorly suited for real-world decision-making workflows. 

        Customized, AI-based [research and report generation](https://jxnl.co/writing/2024/06/05/predictions-for-the-future-of-rag/#reports-over-rag) workflows are a promising way to address this.

        ## Goal

        Our goal is to build a lightweight, multi-agent system around chat models that customizes the research process.

        `Source Selection` 
        * Users can choose any set of input sources for their research.
          
        `Planning` 
        * Users provide a topic, and the system generates a team of AI analysts, each focusing on one sub-topic.
        * `Human-in-the-loop` will be used to refine these sub-topics before research begins.
          
        `LLM Utilization`
        * Each analyst will conduct in-depth interviews with an expert AI using the selected sources.
        * The interview will be a multi-turn conversation to extract detailed insights as shown in the [STORM](https://github.com/langchain-ai/langgraph/blob/main/examples/storm/storm.ipynb) paper.
        * These interviews will be captured in a using `sub-graphs` with their internal state. 
           
        `Research Process`
        * Experts will gather information to answer analyst questions in `parallel`.
        * And all interviews will be conducted simultaneously through `map-reduce`.

        `Output Format` 
        * The gathered insights from each interview will be synthesized into a final report.
        * We'll use customizable prompts for the report, allowing for a flexible output format. 

        ![Screenshot 2024-08-26 at 7.26.33 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb164d61c93d48e604091_research-assistant1.png)
        """
    )
    return


@app.cell
def __():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-stderr
    # %pip install --quiet -U langgraph langchain_openai langchain_community langchain_core tavily-python wikipedia
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Setup
        """
    )
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
def __():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return ChatOpenAI, llm


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
        ## Generate Analysts: Human-In-The-Loop

        Create analysts and review them using human-in-the-loop.
        """
    )
    return


@app.cell
def __():
    from typing import List
    from typing_extensions import TypedDict
    from pydantic import BaseModel, Field

    class Analyst(BaseModel):
        affiliation: str = Field(
            description="Primary affiliation of the analyst.",
        )
        name: str = Field(
            description="Name of the analyst."
        )
        role: str = Field(
            description="Role of the analyst in the context of the topic.",
        )
        description: str = Field(
            description="Description of the analyst focus, concerns, and motives.",
        )
        @property
        def persona(self) -> str:
            return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

    class Perspectives(BaseModel):
        analysts: List[Analyst] = Field(
            description="Comprehensive list of analysts with their roles and affiliations.",
        )

    class GenerateAnalystsState(TypedDict):
        topic: str # Research topic
        max_analysts: int # Number of analysts
        human_analyst_feedback: str # Human feedback
        analysts: List[Analyst] # Analyst asking questions
    return (
        Analyst,
        BaseModel,
        Field,
        GenerateAnalystsState,
        List,
        Perspectives,
        TypedDict,
    )


@app.cell
def __(GenerateAnalystsState, Perspectives, builder, llm, memory):
    from IPython.display import Image, display
    from langgraph.graph import START, END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    analyst_instructions = 'You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:\n\n1. First, review the research topic:\n{topic}\n        \n2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: \n        \n{human_analyst_feedback}\n    \n3. Determine the most interesting themes based upon documents and / or feedback above.\n                    \n4. Pick the top {max_analysts} themes.\n\n5. Assign one analyst to each theme.'

    def create_analysts(state: GenerateAnalystsState):
        """ Create analysts """
        topic = state['topic']
        max_analysts = state['max_analysts']
        human_analyst_feedback = state.get('human_analyst_feedback', '')
        structured_llm = llm.with_structured_output(Perspectives)
        system_message = analyst_instructions.format(topic=topic, human_analyst_feedback=human_analyst_feedback, max_analysts=max_analysts)
        analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content='Generate the set of analysts.')])
        return {'analysts': analysts.analysts}

    def human_feedback(state: GenerateAnalystsState):
        """ No-op node that should be interrupted on """
        pass

    def should_continue(state: GenerateAnalystsState):
        """ Return the next node to execute """
        human_analyst_feedback = state.get('human_analyst_feedback', None)
        if human_analyst_feedback:
            return 'create_analysts'
        return END
    _builder = StateGraph(GenerateAnalystsState)
    _builder.add_node('create_analysts', create_analysts)
    _builder.add_node('human_feedback', human_feedback)
    _builder.add_edge(START, 'create_analysts')
    _builder.add_edge('create_analysts', 'human_feedback')
    _builder.add_conditional_edges('human_feedback', should_continue, ['create_analysts', END])
    _memory = MemorySaver()
    graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
    display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    return (
        AIMessage,
        END,
        HumanMessage,
        Image,
        MemorySaver,
        START,
        StateGraph,
        SystemMessage,
        analyst_instructions,
        create_analysts,
        display,
        graph,
        human_feedback,
        should_continue,
    )


@app.cell
def __(event, graph):
    _max_analysts = 3
    topic = 'The benefits of adopting LangGraph as an agent framework'
    thread = {'configurable': {'thread_id': '1'}}
    for _event in graph.stream({'topic': topic, 'max_analysts': _max_analysts}, thread, stream_mode='values'):
        analysts = event.get('analysts', '')
        if analysts:
            for _analyst in analysts:
                print(f'Name: {_analyst.name}')
                print(f'Affiliation: {_analyst.affiliation}')
                print(f'Role: {_analyst.role}')
                print(f'Description: {_analyst.description}')
                print('-' * 50)
    return analysts, thread, topic


@app.cell
def __(graph, thread):
    # Get state and look at next node
    state = graph.get_state(thread)
    state.next
    return (state,)


@app.cell
def __(graph, thread):
    # We now update the state as if we are the human_feedback node
    graph.update_state(thread, {"human_analyst_feedback": 
                                "Add in someone from a startup to add an entrepreneur perspective"}, as_node="human_feedback")
    return


@app.cell
def __(event, graph, thread):
    for _event in graph.stream(None, thread, stream_mode='values'):
        analysts_1 = event.get('analysts', '')
        if analysts_1:
            for _analyst in analysts_1:
                print(f'Name: {_analyst.name}')
                print(f'Affiliation: {_analyst.affiliation}')
                print(f'Role: {_analyst.role}')
                print(f'Description: {_analyst.description}')
                print('-' * 50)
    return (analysts_1,)


@app.cell
def __(graph, thread):
    # If we are satisfied, then we simply supply no feedback
    further_feedack = None
    graph.update_state(thread, {"human_analyst_feedback": 
                                further_feedack}, as_node="human_feedback")
    return (further_feedack,)


@app.cell
def __(event, graph, thread):
    for _event in graph.stream(None, thread, stream_mode='updates'):
        print('--Node--')
        _node_name = next(iter(event.keys()))
        print(_node_name)
    return


@app.cell
def __(graph, thread):
    final_state = graph.get_state(thread)
    analysts_2 = final_state.values.get('analysts')
    return analysts_2, final_state


@app.cell
def __(final_state):
    final_state.next
    return


@app.cell
def __(analysts_2):
    for _analyst in analysts_2:
        print(f'Name: {_analyst.name}')
        print(f'Affiliation: {_analyst.affiliation}')
        print(f'Role: {_analyst.role}')
        print(f'Description: {_analyst.description}')
        print('-' * 50)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Conduct Interview

        ### Generate Question

        The analyst will ask questions to the expert.
        """
    )
    return


@app.cell
def __(Analyst, BaseModel, Field):
    import operator
    from typing import  Annotated
    from langgraph.graph import MessagesState

    class InterviewState(MessagesState):
        max_num_turns: int # Number turns of conversation
        context: Annotated[list, operator.add] # Source docs
        analyst: Analyst # Analyst asking questions
        interview: str # Interview transcript
        sections: list # Final key we duplicate in outer state for Send() API

    class SearchQuery(BaseModel):
        search_query: str = Field(None, description="Search query for retrieval.")
    return Annotated, InterviewState, MessagesState, SearchQuery, operator


@app.cell
def __(InterviewState, SystemMessage, llm):
    question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

    Your goal is boil down to interesting and specific insights related to your topic.

    1. Interesting: Insights that people will find surprising or non-obvious.
            
    2. Specific: Insights that avoid generalities and include specific examples from the expert.

    Here is your topic of focus and set of goals: {goals}
            
    Begin by introducing yourself using a name that fits your persona, and then ask your question.

    Continue to ask questions to drill down and refine your understanding of the topic.
            
    When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

    Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

    def generate_question(state: InterviewState):
        """ Node to generate a question """

        # Get state
        analyst = state["analyst"]
        messages = state["messages"]

        # Generate question 
        system_message = question_instructions.format(goals=analyst.persona)
        question = llm.invoke([SystemMessage(content=system_message)]+messages)
            
        # Write messages to state
        return {"messages": [question]}
    return generate_question, question_instructions


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Generate Answer: Parallelization

        The expert will gather information from multiple sources in parallel to answer questions.

        For example, we can use:

        * Specific web sites e.g., via [`WebBaseLoader`](https://python.langchain.com/v0.2/docs/integrations/document_loaders/web_base/)
        * Indexed documents e.g., via [RAG](https://python.langchain.com/v0.2/docs/tutorials/rag/)
        * Web search
        * Wikipedia search

        You can try different web search tools, like [Tavily](https://tavily.com/).
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
def __():
    # Web search tool
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_search = TavilySearchResults(max_results=3)
    return TavilySearchResults, tavily_search


@app.cell
def __():
    # Wikipedia search tool
    from langchain_community.document_loaders import WikipediaLoader
    return (WikipediaLoader,)


@app.cell
def __(mo):
    mo.md(
        r"""
        Now, we create nodes to search the web and wikipedia.

        We'll also create a node to answer analyst questions.

        Finally, we'll create nodes to save the full interview and to write a summary ("section") of the interview.
        """
    )
    return


@app.cell
def __(
    AIMessage,
    END,
    HumanMessage,
    Image,
    InterviewState,
    MemorySaver,
    START,
    SearchQuery,
    StateGraph,
    SystemMessage,
    WikipediaLoader,
    display,
    generate_question,
    llm,
    memory,
    tavily_search,
):
    from langchain_core.messages import get_buffer_string
    search_instructions = SystemMessage(content=f'You will be given a conversation between an analyst and an expert. \n\nYour goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.\n        \nFirst, analyze the full conversation.\n\nPay particular attention to the final question posed by the analyst.\n\nConvert this final question into a well-structured web search query')

    def search_web(state: InterviewState):
        """ Retrieve docs from web search """
        structured_llm = llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions] + state['messages'])
        search_docs = tavily_search.invoke(search_query.search_query)
        formatted_search_docs = '\n\n---\n\n'.join([f'<Document href="{doc['url']}"/>\n{doc['content']}\n</Document>' for doc in search_docs])
        return {'context': [formatted_search_docs]}

    def search_wikipedia(state: InterviewState):
        """ Retrieve docs from wikipedia """
        structured_llm = llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions] + state['messages'])
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
        formatted_search_docs = '\n\n---\n\n'.join([f'<Document source="{doc.metadata['source']}" page="{doc.metadata.get('page', '')}"/>\n{doc.page_content}\n</Document>' for doc in search_docs])
        return {'context': [formatted_search_docs]}
    answer_instructions = 'You are an expert being interviewed by an analyst.\n\nHere is analyst area of focus: {goals}. \n        \nYou goal is to answer a question posed by the interviewer.\n\nTo answer question, use this context:\n        \n{context}\n\nWhen answering questions, follow these guidelines:\n        \n1. Use only the information provided in the context. \n        \n2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.\n\n3. The context contain sources at the topic of each individual document.\n\n4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. \n\n5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc\n        \n6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>\' then just list: \n        \n[1] assistant/docs/llama3_1.pdf, page 7 \n        \nAnd skip the addition of the brackets as well as the Document source preamble in your citation.'

    def generate_answer(state: InterviewState):
        """ Node to answer a question """
        analyst = state['analyst']
        messages = state['messages']
        context = state['context']
        system_message = answer_instructions.format(goals=analyst.persona, context=context)
        answer = llm.invoke([SystemMessage(content=system_message)] + messages)
        answer.name = 'expert'
        return {'messages': [answer]}

    def save_interview(state: InterviewState):
        """ Save interviews """
        messages = state['messages']
        interview = get_buffer_string(messages)
        return {'interview': interview}

    def route_messages(state: InterviewState, name: str='expert'):
        """ Route between question and answer """
        messages = state['messages']
        max_num_turns = state.get('max_num_turns', 2)
        num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
        if num_responses >= max_num_turns:
            return 'save_interview'
        last_question = messages[-2]
        if 'Thank you so much for your help' in last_question.content:
            return 'save_interview'
        return 'ask_question'
    section_writer_instructions = 'You are an expert technical writer. \n            \nYour task is to create a short, easily digestible section of a report based on a set of source documents.\n\n1. Analyze the content of the source documents: \n- The name of each source document is at the start of the document, with the <Document tag.\n        \n2. Create a report structure using markdown formatting:\n- Use ## for the section title\n- Use ### for sub-section headers\n        \n3. Write the report following this structure:\na. Title (## header)\nb. Summary (### header)\nc. Sources (### header)\n\n4. Make your title engaging based upon the focus area of the analyst: \n{focus}\n\n5. For the summary section:\n- Set up summary with general background / context related to the focus area of the analyst\n- Emphasize what is novel, interesting, or surprising about insights gathered from the interview\n- Create a numbered list of source documents, as you use them\n- Do not mention the names of interviewers or experts\n- Aim for approximately 400 words maximum\n- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents\n        \n6. In the Sources section:\n- Include all sources used in your report\n- Provide full links to relevant websites or specific document paths\n- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.\n- It will look like:\n\n### Sources\n[1] Link or Document name\n[2] Link or Document name\n\n7. Be sure to combine sources. For example this is not correct:\n\n[3] https://ai.meta.com/blog/meta-llama-3-1/\n[4] https://ai.meta.com/blog/meta-llama-3-1/\n\nThere should be no redundant sources. It should simply be:\n\n[3] https://ai.meta.com/blog/meta-llama-3-1/\n        \n8. Final review:\n- Ensure the report follows the required structure\n- Include no preamble before the title of the report\n- Check that all guidelines have been followed'

    def write_section(state: InterviewState):
        """ Node to answer a question """
        interview = state['interview']
        context = state['context']
        analyst = state['analyst']
        system_message = section_writer_instructions.format(focus=analyst.description)
        section = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f'Use this source to write your section: {context}')])
        return {'sections': [section.content]}
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node('ask_question', generate_question)
    interview_builder.add_node('search_web', search_web)
    interview_builder.add_node('search_wikipedia', search_wikipedia)
    interview_builder.add_node('answer_question', generate_answer)
    interview_builder.add_node('save_interview', save_interview)
    interview_builder.add_node('write_section', write_section)
    interview_builder.add_edge(START, 'ask_question')
    interview_builder.add_edge('ask_question', 'search_web')
    interview_builder.add_edge('ask_question', 'search_wikipedia')
    interview_builder.add_edge('search_web', 'answer_question')
    interview_builder.add_edge('search_wikipedia', 'answer_question')
    interview_builder.add_conditional_edges('answer_question', route_messages, ['ask_question', 'save_interview'])
    interview_builder.add_edge('save_interview', 'write_section')
    interview_builder.add_edge('write_section', END)
    _memory = MemorySaver()
    interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name='Conduct Interviews')
    display(Image(interview_graph.get_graph().draw_mermaid_png()))
    return (
        answer_instructions,
        generate_answer,
        get_buffer_string,
        interview_builder,
        interview_graph,
        route_messages,
        save_interview,
        search_instructions,
        search_web,
        search_wikipedia,
        section_writer_instructions,
        write_section,
    )


@app.cell
def __(analysts_2):
    analysts_2[0]
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Here, we run the interview passing an index of the llama3.1 paper, which is related to our topic.
        """
    )
    return


@app.cell
def __(HumanMessage, analysts_2, interview_graph, topic):
    from IPython.display import Markdown
    messages = [HumanMessage(f'So you said you were writing an article on {topic}?')]
    thread_1 = {'configurable': {'thread_id': '1'}}
    interview = interview_graph.invoke({'analyst': analysts_2[0], 'messages': messages, 'max_num_turns': 2}, thread_1)
    Markdown(interview['sections'][0])
    return Markdown, interview, messages, thread_1


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Parallelze interviews: Map-Reduce

        We parallelize the interviews via the `Send()` API, a map step.

        We combine them into the report body in a reduce step.

        ### Finalize

        We add a final step to write an intro and conclusion to the final report.
        """
    )
    return


@app.cell
def __(Analyst, Annotated_1, List_1, TypedDict, operator):
    from typing import List, Annotated

    class ResearchGraphState(TypedDict):
        topic: str
        max_analysts: int
        human_analyst_feedback: str
        analysts: List_1[Analyst]
        sections: Annotated_1[list, operator.add]
        introduction: str
        content: str
        conclusion: str
        final_report: str
    return Annotated, List, ResearchGraphState


@app.cell
def __(
    END,
    HumanMessage,
    Image,
    MemorySaver,
    ResearchGraphState,
    START,
    StateGraph,
    SystemMessage,
    builder,
    create_analysts,
    display,
    human_feedback,
    interview_builder,
    llm,
    memory,
):
    from langgraph.constants import Send

    def initiate_all_interviews(state: ResearchGraphState):
        """ This is the "map" step where we run each interview sub-graph using Send API """
        human_analyst_feedback = state.get('human_analyst_feedback')
        if human_analyst_feedback:
            return 'create_analysts'
        else:
            topic = state['topic']
            return [Send('conduct_interview', {'analyst': analyst, 'messages': [HumanMessage(content=f'So you said you were writing an article on {topic}?')]}) for analyst in state['analysts']]
    report_writer_instructions = 'You are a technical writer creating a report on this overall topic: \n\n{topic}\n    \nYou have a team of analysts. Each analyst has done two things: \n\n1. They conducted an interview with an expert on a specific sub-topic.\n2. They write up their finding into a memo.\n\nYour task: \n\n1. You will be given a collection of memos from your analysts.\n2. Think carefully about the insights from each memo.\n3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. \n4. Summarize the central points in each memo into a cohesive single narrative.\n\nTo format your report:\n \n1. Use markdown formatting. \n2. Include no pre-amble for the report.\n3. Use no sub-heading. \n4. Start your report with a single title header: ## Insights\n5. Do not mention any analyst names in your report.\n6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].\n7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.\n8. List your sources in order and do not repeat.\n\n[1] Source 1\n[2] Source 2\n\nHere are the memos from your analysts to build your report from: \n\n{context}'

    def write_report(state: ResearchGraphState):
        sections = state['sections']
        topic = state['topic']
        formatted_str_sections = '\n\n'.join([f'{section}' for section in sections])
        system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
        report = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f'Write a report based upon these memos.')])
        return {'content': report.content}
    intro_conclusion_instructions = 'You are a technical writer finishing a report on {topic}\n\nYou will be given all of the sections of the report.\n\nYou job is to write a crisp and compelling introduction or conclusion section.\n\nThe user will instruct you whether to write the introduction or conclusion.\n\nInclude no pre-amble for either section.\n\nTarget around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.\n\nUse markdown formatting. \n\nFor your introduction, create a compelling title and use the # header for the title.\n\nFor your introduction, use ## Introduction as the section header. \n\nFor your conclusion, use ## Conclusion as the section header.\n\nHere are the sections to reflect on for writing: {formatted_str_sections}'

    def write_introduction(state: ResearchGraphState):
        sections = state['sections']
        topic = state['topic']
        formatted_str_sections = '\n\n'.join([f'{section}' for section in sections])
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
        intro = llm.invoke([instructions] + [HumanMessage(content=f'Write the report introduction')])
        return {'introduction': intro.content}

    def write_conclusion(state: ResearchGraphState):
        sections = state['sections']
        topic = state['topic']
        formatted_str_sections = '\n\n'.join([f'{section}' for section in sections])
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
        conclusion = llm.invoke([instructions] + [HumanMessage(content=f'Write the report conclusion')])
        return {'conclusion': conclusion.content}

    def finalize_report(state: ResearchGraphState):
        """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
        content = state['content']
        if content.startswith('## Insights'):
            content = content.strip('## Insights')
        if '## Sources' in content:
            try:
                content, sources = content.split('\n## Sources\n')
            except:
                sources = None
        else:
            sources = None
        final_report = state['introduction'] + '\n\n---\n\n' + content + '\n\n---\n\n' + state['conclusion']
        if sources is not None:
            final_report = final_report + ('\n\n## Sources\n' + sources)
        return {'final_report': final_report}
    _builder = StateGraph(ResearchGraphState)
    _builder.add_node('create_analysts', create_analysts)
    _builder.add_node('human_feedback', human_feedback)
    _builder.add_node('conduct_interview', interview_builder.compile())
    _builder.add_node('write_report', write_report)
    _builder.add_node('write_introduction', write_introduction)
    _builder.add_node('write_conclusion', write_conclusion)
    _builder.add_node('finalize_report', finalize_report)
    _builder.add_edge(START, 'create_analysts')
    _builder.add_edge('create_analysts', 'human_feedback')
    _builder.add_conditional_edges('human_feedback', initiate_all_interviews, ['create_analysts', 'conduct_interview'])
    _builder.add_edge('conduct_interview', 'write_report')
    _builder.add_edge('conduct_interview', 'write_introduction')
    _builder.add_edge('conduct_interview', 'write_conclusion')
    _builder.add_edge(['write_conclusion', 'write_report', 'write_introduction'], 'finalize_report')
    _builder.add_edge('finalize_report', END)
    _memory = MemorySaver()
    graph_1 = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
    display(Image(graph_1.get_graph(xray=1).draw_mermaid_png()))
    return (
        Send,
        finalize_report,
        graph_1,
        initiate_all_interviews,
        intro_conclusion_instructions,
        report_writer_instructions,
        write_conclusion,
        write_introduction,
        write_report,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        Let's ask an open-ended question about LangGraph.
        """
    )
    return


@app.cell
def __(event, graph_1):
    _max_analysts = 3
    topic_1 = 'The benefits of adopting LangGraph as an agent framework'
    thread_2 = {'configurable': {'thread_id': '1'}}
    for _event in graph_1.stream({'topic': topic_1, 'max_analysts': _max_analysts}, thread_2, stream_mode='values'):
        analysts_3 = event.get('analysts', '')
        if analysts_3:
            for _analyst in analysts_3:
                print(f'Name: {_analyst.name}')
                print(f'Affiliation: {_analyst.affiliation}')
                print(f'Role: {_analyst.role}')
                print(f'Description: {_analyst.description}')
                print('-' * 50)
    return analysts_3, thread_2, topic_1


@app.cell
def __(graph_1, thread_2):
    graph_1.update_state(thread_2, {'human_analyst_feedback': 'Add in the CEO of gen ai native startup'}, as_node='human_feedback')
    return


@app.cell
def __(event, graph_1, thread_2):
    for _event in graph_1.stream(None, thread_2, stream_mode='values'):
        analysts_4 = event.get('analysts', '')
        if analysts_4:
            for _analyst in analysts_4:
                print(f'Name: {_analyst.name}')
                print(f'Affiliation: {_analyst.affiliation}')
                print(f'Role: {_analyst.role}')
                print(f'Description: {_analyst.description}')
                print('-' * 50)
    return (analysts_4,)


@app.cell
def __(graph_1, thread_2):
    graph_1.update_state(thread_2, {'human_analyst_feedback': None}, as_node='human_feedback')
    return


@app.cell
def __(event, graph_1, thread_2):
    for _event in graph_1.stream(None, thread_2, stream_mode='updates'):
        print('--Node--')
        _node_name = next(iter(event.keys()))
        print(_node_name)
    return


@app.cell
def __(Markdown, graph_1, thread_2):
    final_state_1 = graph_1.get_state(thread_2)
    report = final_state_1.values.get('final_report')
    Markdown(report)
    return final_state_1, report


@app.cell
def __(mo):
    mo.md(
        r"""
        We can look at the trace:

        https://smith.langchain.com/public/2933a7bb-bcef-4d2d-9b85-cc735b22ca0c/r
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

