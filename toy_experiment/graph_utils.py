from dotenv import load_dotenv
load_dotenv()
import os
import openai
from typing import List, Type, Optional, Annotated, Sequence, Literal, TypedDict
from typing_extensions import TypedDict
from pydantic import BaseModel, create_model, field_validator, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import ToolNode, InjectedState # Good docs / source code https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langworkflow/prebuilt/tool_node.py
from langgraph.graph import StateGraph, START, END
import operator
import functools
from PIL import Image
import io
from copy import deepcopy
from pathlib import Path
from pathlib import Path
import importlib
import env
import utils
import json
importlib.reload(env)
importlib.reload(utils)
from datetime import datetime


def create_graph(env_path : Path, llm: ChatOpenAI, max_inference : int, graph_type : str, prompting_method : str = 'default', trial_name = 'testing', custom_path : Path = Path()):
    envr = env.env.from_json(env_path)
    if custom_path == Path():
        path = (env_path/ '..' / '..' / '..' / 'logs' / envr.config['edition'] / graph_type / llm.model_name / prompting_method / trial_name).resolve()
        if not os.path.exists(path):
            os.makedirs(path)
        drs = os.listdir(path)
        # path = path / f'run_{len(drs)}'
        # os.makedirs(path)
        filepath = path
    else:
        filepath = custom_path

    save_dict = {
        'env_config' : json.load(open(env_path)), 
        'run_config' : {'graph_type' : graph_type, 'prompting_method' : prompting_method, 'max_inference' : max_inference, 'run_date' : tuple(datetime.now().timetuple())}
        }

    class AgentsInterface(BaseTool):
        name: str = "AgentsInterface"
        description: str = "How you instruct the agents to follow a particular path. All agents should recieve a path according to the argument description in this singular tool call."
        args_schema: Type[BaseModel] = envr.generate_env_tool() # ALTER THIS
        return_direct: bool = True


        def _run(
            self, **kwargs
        ) -> str:
            """Use the tool."""
            # The "$xtra" prefix on a kwarg means it isn't an agent path and serves another purpose
            # Ok i was wrong about the $xtra stuff bc it would have to go through the pydantic interface which would make it an arg for the LLM
            # What i will do it somehow set the env value in the state, and then pass the state to the tool
            # IDk man check this out https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools
            # Might not work, good luck!!
            print(kwargs)

            agent_paths = {k : [(p.x, p.y, p.z)for p in v] for k, v in kwargs.items() if k != 'env'} # Converts point object to tuple
            print(agent_paths)
            print(kwargs.get('env', 'noenv'))
            run_env : env.env = kwargs.get('env')
            plan, feedback, has_feedback = run_env.tool_call_backend(agent_paths)
            print(f'Plan: {plan}')
            print(f'Feedback: {feedback}')
            print(f'Has Feedback: {has_feedback}')
            return run_env.prompt(method = 'default', feedback_plan = feedback)[2], not has_feedback
    

    tools = [AgentsInterface()]
    tool_node = functools.partial(gen_tool_node, tools=tools)

    builder = StateGraph(State)
    builder.add_node('call_tool', tool_node)

    graph_type_list = ['central_agent_1', 'multi_agent_1']
    if graph_type.lower() == 'central_agent_1':


        central_agent = create_agent(
            llm, 
            tools,
            system_message=""
        )

        central_agent_node = functools.partial(agent_node, agent=central_agent, name='central_agent', max_inference=max_inference)

        def router(state) -> Literal['call_tool', '__end__', 'continue']:
            messages = state['messages']
            last_message = messages[-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return 'call_tool'
            # You're normally supposed to add some condition to end and otherwise continue, but this is just for testing
            elif state['success'] or state['inference_count'] > max_inference:
                save_dict['final_vals'] = end_vals(state)
                json.dump(save_dict, open(filepath / f'env_{envr.config["env_num"]}.json', 'w'))
                return '__end__'
            else:
                return 'continue'
        
        builder.add_node('central_agent', central_agent_node)
        builder.add_node('no_tool', no_tool)
        builder.add_node('end_node', end_node)
        builder.add_edge('end_node' , END)


        builder.add_conditional_edges(
            'central_agent',
            router,
            {'continue' : 'no_tool','call_tool' : 'call_tool', '__end__' : 'end_node',}
        )

        builder.add_edge('no_tool', 'central_agent')

        builder.add_conditional_edges(
            'call_tool',
            router,
            {'continue' : 'central_agent', '__end__' : 'end_node'}
        )

        builder.add_edge(START, 'central_agent')

    elif graph_type.lower() == 'multi_agent_1':
        
        agents = {name : create_agent(llm, tools,system_message=f"You are Agent {name}.") for name in envr.agents}


    else:
        raise ValueError(f'Incompatible graph type. Graph type must be one of: {",".join(graph_type_list)}')
    


    graph = builder.compile() # Add memory / persistence / checkpointer?

    image_bytes = graph.get_graph().draw_mermaid_png()  
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    image.save(filepath / 'graph_diagram.png', 'PNG')
    image_stream.close()

    msg = HumanMessage(content=envr.prompt(method=prompting_method)[1])

    events = graph.astream({'messages' : [msg], 'inference_count' : 0 , 'env' : envr, 'success' : False})

    return events
    
    






def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Plan paths for agents to navigate a 3D grid to reach their respective goals and avoid collision."
                " You are given:"
                " 1) a list of obstacle coordinates (x, y, z): locations of the obstacle grid cells, agents must avoid them."
                " 2) a list of [([name], [init], [goal]) tuples], [init] and [goal] are 3D coordinates of the initial position and goal position of agent named [name]."
                " 3) a previous plan, if any, and why it failed. Analyze this information and re-plan a collision-free path."
                " How to plan a <path>:"
                " 1) Make sure each path does not touch any obstacle or another agent."
                " 2) Create a set of points for each agent to go from their init coordinates to their goal coordinates."
                " 3) Make sure the coordinates are exactly one step away from each other, in one direction. Note - you may only move in one direction at a time."
                " Example of a <path>: [{{'x' : 1, 'y' : 1, 'z' : 1}}, {{'x' : 1, 'y' : 1, 'z' : 2}}, {{'x' : 1, 'y' : 1, 'z' : 3}},...]"
                " Output Instruction: Use the Agent Interface tool provided to output your final plan for the agents. Only one tool call is required for all agents. \n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    inference_count: int
    env : dict
    success : bool = False


def agent_node(state, agent, name, max_inference):
    if state['inference_count'] > max_inference:
        return {'inference_count' : state['inference_count'] + 1,}
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), sender=name)
    return {
        "messages": [result],
        "sender": name,
        'inference_count' : state['inference_count'] + 1,
    }


def no_tool(state):
    msg = HumanMessage(content='No tool call found - make sure the tool interfaces are used as properly described.')
    return {
        'messages' : [msg],
        'sender' : 'human'
    }


def gen_tool_node(state, tools):
        result = []
        tools_by_name = {tool.name: tool for tool in tools}
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            call = deepcopy(tool_call)
            if tool_call['name'] == 'AgentsInterface':
                call['args']['env'] = state['env']
            observation = tool.invoke(call["args"])
            result.append(ToolMessage(content=observation[0], sender=call['name'], tool_call_id=call["id"]))
        return {"messages": result, 'success' : observation[1]}



def end_vals(state: State):
    return {
        'messages' : [m.to_json() for m in state['messages']],
        'inference_count' : state['inference_count'],
        'success' : state['success'],
    }


def end_node(state):
    return {'success' : state['success'], 'inference_count' : state['inference_count']}