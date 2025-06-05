import operator
import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from google.api_core.client_options import ClientOptions # Added import

from ..tools.web_research_tool import WebResearchTool

# Load environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

google_api_key = os.getenv("GEMINI_API_KEY") # Google Search also uses this often, or a separate GOOGLE_API_KEY
google_cse_id = os.getenv("GOOGLE_CSE_ID")
if not google_cse_id:
    raise ValueError("GOOGLE_CSE_ID environment variable not set for WebResearchTool.")


# Helper function to get LLM instance with custom endpoint
def get_llm_instance(model_name: str, api_key: str, temperature: float, top_p: float) -> ChatGoogleGenerativeAI:
    """
    Creates an instance of ChatGoogleGenerativeAI, configured with a custom endpoint if specified.
    """
    custom_gemini_endpoint_url = os.getenv("CUSTOM_GEMINI_API_ENDPOINT")
    model_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
    }

    if custom_gemini_endpoint_url:
        # api_endpoint for ClientOptions typically expects 'hostname/path' or just 'hostname'
        # We strip the scheme as the SDK usually handles it.
        api_endpoint_val = custom_gemini_endpoint_url
        if api_endpoint_val.startswith("https://"):
            api_endpoint_val = api_endpoint_val[len("https://"):]
        elif api_endpoint_val.startswith("http://"):
            api_endpoint_val = api_endpoint_val[len("http://"):]
        
        client_opts = ClientOptions(api_endpoint=api_endpoint_val)
        model_kwargs["client_options"] = client_opts
        print(f"INFO: Using custom Gemini API endpoint '{api_endpoint_val}' for model {model_name}")
    else:
        print(f"INFO: Using default Gemini API endpoint for model {model_name}")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        **model_kwargs
    )

# Configure WebResearchTool with potential custom endpoint for its LLM (if used internally)
web_research_llm_kwargs = {"temperature": 0.0, "top_p": 0.9}
custom_gemini_endpoint_url_for_tool = os.getenv("CUSTOM_GEMINI_API_ENDPOINT")
if custom_gemini_endpoint_url_for_tool:
    api_endpoint_val_for_tool = custom_gemini_endpoint_url_for_tool
    if api_endpoint_val_for_tool.startswith("https://"):
        api_endpoint_val_for_tool = api_endpoint_val_for_tool[len("https://"):]
    elif api_endpoint_val_for_tool.startswith("http://"):
        api_endpoint_val_for_tool = api_endpoint_val_for_tool[len("http://"):]
    
    client_opts_for_tool = ClientOptions(api_endpoint=api_endpoint_val_for_tool)
    web_research_llm_kwargs["client_options"] = client_opts_for_tool # Assuming WebResearchTool passes client_options if it uses an LLM
    print(f"INFO: Configuring WebResearchTool LLM_KWARGS with custom Gemini API endpoint: {api_endpoint_val_for_tool}")
else:
    print(f"INFO: WebResearchTool using default Gemini API endpoint for its internal LLM (if any).")

research_tool = WebResearchTool(
    google_api_key=google_api_key, # This is for Google Search API, not Gemini directly
    google_cse_id=google_cse_id,
    llm_kwargs=web_research_llm_kwargs, # These kwargs are for any LLM used *within* WebResearchTool (e.g. for summarization)
)


# Define AgentState
class AgentState(TypedDict):
    input: str
    messages: Annotated[list[AnyMessage], operator.add]
    queries_generated_count: int
    initial_search_query_count: int
    max_research_loops: int
    current_queries: list[str]
    research_summary: Annotated[list[dict], operator.add]
    final_answer: str
    reasoning_model: str # Added to carry the model choice from input


# Define Pydantic models for structured output
class Queries(BaseModel):
    query_list: list[str] = Field(
        description="Comprehensive list of search engine queries to answer the user's question."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the information gathered is sufficient to answer the user's question."
    )
    follow_up_queries: list[str] = Field(
        description="List of follow-up queries if information is not sufficient. Empty list if sufficient."
    )


class FinalAnswer(BaseModel):
    answer: str = Field(
        description="The final answer to the user's question, synthesized from the research findings, with citations."
    )


# Define agent nodes
def generate_initial_queries_node(state: AgentState):
    print("---GENERATE INITIAL QUERIES---")
    prompt_template = PromptTemplate.from_template(
        """You are a research assistant. Your task is to generate a list of {query_count} distinct search queries that will help answer the user's question: {question}.
        Focus on creating queries that cover different aspects of the question and are likely to yield relevant and comprehensive results.
        Do not generate too similar queries. If the user asks for a very specific fact, one query might be enough.
        """
    )
    reasoning_model_name = state["reasoning_model"]
    reasoning_llm = get_llm_instance(
        model_name=reasoning_model_name, 
        api_key=gemini_api_key, 
        temperature=0.0, 
        top_p=0.9
    )
    
    generate_queries_chain = prompt_template | reasoning_llm.with_structured_output(
        Queries
    )
    query_count = state["initial_search_query_count"]
    queries_output = generate_queries_chain.invoke(
        {"question": state["input"], "query_count": query_count}
    )
    return {
        "current_queries": queries_output.query_list,
        "queries_generated_count": len(queries_output.query_list),
        "messages": [("ai", f"Generated queries: {queries_output.query_list}")],
    }


def web_research_node(state: AgentState):
    print("---WEB RESEARCH---")
    all_results = []
    for query in state["current_queries"]:
        print(f"Searching for query: {query}")
        # research_tool.run returns a list of dicts, each with 'url' and 'content'
        # We need to adapt this if the structure is different. Let's assume it's a string or list of strings.
        sources = research_tool.run(query) # This uses Google Search API
        all_results.append({"query": query, "sources_gathered": sources})

    return {"research_summary": all_results}


def reflect_node(state: AgentState):
    print("---REFLECT ON RESULTS---")
    prompt_template = PromptTemplate.from_template(
        """You are a research assistant. You have gathered the following information for the user's question: {question}.
        Research Summary:
        {research_summary}

        Based on this information, is it sufficient to provide a comprehensive answer?
        If not, what specific follow-up search queries are needed to fill the gaps?
        Generate a maximum of {query_count} follow-up queries.
        """
    )
    reflection_model_name = state["reasoning_model"] # Using reasoning_model for reflection as well
    reflection_llm = get_llm_instance(
        model_name=reflection_model_name,
        api_key=gemini_api_key,
        temperature=0.0,
        top_p=0.9
    )
    reflect_chain = prompt_template | reflection_llm.with_structured_output(Reflection)
    query_count = state["initial_search_query_count"] # Using initial count for follow-ups as well for simplicity
    
    # Format research_summary for the prompt
    formatted_summary = ""
    for item in state["research_summary"]:
        formatted_summary += f"Query: {item['query']}\nSources:\n"
        if isinstance(item["sources_gathered"], list):
            for source in item["sources_gathered"][:3]: # Show first 3 sources per query
                 if isinstance(source, dict) and 'content' in source and 'url' in source:
                    content_snippet = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                    formatted_summary += f"  - {source['url']}: {content_snippet}\n"
                 else: # If sources are not dicts with url/content, stringify them
                    formatted_summary += f"  - {str(source)}\n"

        else:
            formatted_summary += f"  - {str(item['sources_gathered'])}\n"
        formatted_summary += "\n"

    reflection_output = reflect_chain.invoke(
        {
            "question": state["input"],
            "research_summary": formatted_summary,
            "query_count": query_count,
        }
    )
    return {
        "current_queries": reflection_output.follow_up_queries,
        "queries_generated_count": state["queries_generated_count"]
        + len(reflection_output.follow_up_queries),
        "messages": [("ai", f"Reflection: Sufficient: {reflection_output.is_sufficient}, Follow-up: {reflection_output.follow_up_queries}")]
    }


def generate_final_answer_node(state: AgentState):
    print("---GENERATE FINAL ANSWER---")
    prompt_template = PromptTemplate.from_template(
        """You are a research assistant. Based on the extensive research conducted, synthesize a comprehensive answer to the user's question: {question}.
        Your answer should be well-structured, informative, and directly address the question.
        Where possible, cite the sources used from the research summary provided. Format citations like [Source URL].

        Research Summary:
        {research_summary}
        """
    )
    final_answer_model_name = state["reasoning_model"] # Using reasoning_model for final answer
    answer_llm = get_llm_instance(
        model_name=final_answer_model_name,
        api_key=gemini_api_key,
        temperature=0.7, # Slightly higher temp for more natural answer
        top_p=0.9
    )
    
    final_answer_chain = prompt_template | answer_llm.with_structured_output(
        FinalAnswer
    )

    # Format research_summary for the prompt, including URLs for citation
    formatted_summary_for_answer = ""
    for item in state["research_summary"]:
        formatted_summary_for_answer += f"Information from query '{item['query']}':\n"
        if isinstance(item["sources_gathered"], list):
            for source in item["sources_gathered"]:
                if isinstance(source, dict) and 'content' in source and 'url' in source:
                    content_snippet = source['content'][:300] + "..." if len(source['content']) > 300 else source['content']
                    formatted_summary_for_answer += f"  - Content: {content_snippet} [Source: {source['url']}]\n"
                else:
                    formatted_summary_for_answer += f"  - {str(source)}\n" # Fallback for unexpected source format

        else: # If sources_gathered is not a list (e.g., a string summary)
            formatted_summary_for_answer += f"  - {str(item['sources_gathered'])}\n"
        formatted_summary_for_answer += "\n"


    final_answer_output = final_answer_chain.invoke(
        {"question": state["input"], "research_summary": formatted_summary_for_answer}
    )
    return {
        "final_answer": final_answer_output.answer,
        "messages": [("ai", final_answer_output.answer)],
    }


# Define conditional edges
def should_continue_research(state: AgentState):
    if state["queries_generated_count"] >= (
        state["initial_search_query_count"] * state["max_research_loops"]
    ):  # Max queries limit
        print("---MAX QUERIES REACHED, GENERATING FINAL ANSWER---")
        return "generate_final_answer"
    elif not state["current_queries"]:  # No more follow-up queries from reflection
        print("---NO MORE FOLLOW-UP QUERIES, GENERATING FINAL ANSWER---")
        return "generate_final_answer"
    else:
        print("---CONTINUING RESEARCH---")
        return "web_research"


# Define the graph
workflow = StateGraph(AgentState)

workflow.add_node("generate_query", generate_initial_queries_node)
workflow.add_node("web_research", web_research_node)
workflow.add_node("reflection", reflect_node)
workflow.add_node("finalize_answer", generate_final_answer_node)

workflow.set_entry_point("generate_query")
workflow.add_edge("generate_query", "web_research")
workflow.add_edge("web_research", "reflection")
workflow.add_conditional_edges(
    "reflection",
    should_continue_research,
    {
        "web_research": "web_research",
        "generate_final_answer": "finalize_answer",
    },
)
workflow.add_edge("finalize_answer", END)

graph = workflow.compile()

# Add a stream_events=True configuration for the graph
# This is mainly for the LangGraph SDK to correctly stream intermediate step outputs
# For this example, we'll rely on the print statements and the final AI message for tracing.
# If using LangGraph cloud or specific event streaming, this would be more relevant.
# graph = workflow.compile(stream_events=True) # Example, might need specific setup
