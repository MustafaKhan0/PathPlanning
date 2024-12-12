import numpy as np
from plotly import express as px
from plotly import graph_objects as go
from matplotlib.colors import to_hex, to_rgba
from pathlib import Path
from typing import Union, Annotated
import json
import re
import os
import openai
from typing import List, Optional, Type, Dict, Any


from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, InjectedToolArg
from pydantic import BaseModel, Field, create_model, field_validator, ConfigDict

class Point3D(BaseModel):
    x: int = Field('X Coordinate')
    y: int = Field('Y Coordinate')
    z: int = Field('Z Coordinate')
    
    # Define a validator to ensure each point is a valid 3D point
    @field_validator('x', 'y', 'z')
    def check_coordinates(cls, v):
        if not isinstance(v, int):
            raise ValueError('Coordinate must be an integer')
        return v



class env():
    def __init__(self, agents, obstacles, config, setup_data_path):
        self.agents = agents
        self.obstacles = obstacles
        self.config = config
        self.setup_data_path = setup_data_path
        self.paths = {}


    
    def from_json(path : Path):
        with open(path) as file:
            env_vals = json.load(file)
        
        return env(env_vals['agents'], env_vals['obstacles'], env_vals['config'], (path / '..').resolve())
    
    def from_vals(env_vals):
        return env(env_vals['agents'], env_vals['obstacles'], env_vals['config'], 'N/A')

    
    def generate_env_tool(self):
        '''Interface between Langgraph and the environment made
        '''
        # Tried to do something but doesn't work self.tool_schema = {'interface' : {agent : (List[Point3D], Field(f'A path of 3D points for Agent {agent}, where the list of points is an array and each point is a dictionary.')) for agent in self.agents}}
        self.tool_schema = {'interface' : {agent : (List[Point3D], Field(f'A path of 3D points for Agent {agent}, where the list of points is an array and each point is a dictionary.')) for agent in self.agents} | {'env' : (Annotated[env, InjectedToolArg], Field('env object'))}}
        Interface = create_model('interface', **self.tool_schema['interface'], __config__=ConfigDict(arbitrary_types_allowed=True))
        return Interface


    def visualize(self, model = '', agent_plan = {}):
        '''visualize_env
        Parameters
        ----------
        self
        Returns
        -------
        figure : plotly.graph_objects.Figure
        '''

        fig = go.Figure()

        for obstacle in self.obstacles:
            x,y,z = obstacle['coords']
            fig.add_trace(cubes(obstacle['size'], x, y, z, 'rgba(25,25,25,0.3)'))

        

        palette = px.colors.qualitative.Plotly
        if len(self.agents) > len(px.colors.qualitative.Plotly):
            palette = px.colors.qualitative.Alphabet
        for i, agent_key in enumerate(self.agents):
            init = self.agents[agent_key]['init']
            goal = self.agents[agent_key]['goal']
            if agent_plan:
                path = [init] + agent_plan[agent_key] + [goal]
            elif model:
                path = [init] + self.paths[model][agent_key] + [goal]
            elif self.agents[agent_key]['path']:
                path = [init] + self.agents[agent_key]['path'] + [goal]
            else:
                path = self.agents[agent_key]['path']
            col = [c * 255 for c in to_rgba(palette[i])]
            cols = [
                [col[0]-10, col[1]-10, col[2]-10, col[3]-0.1],
                col,
                [col[0]+10, col[1]+10, col[2]+10, col[3]-0.1],
            ]
            cols = [[str(clamp_color(c)) for c in co[:-1]] + [str(clamp_op(co[3]))] for co in cols]
            colors = [f'rgba({",".join(cole)})' for cole in cols]
            fig.add_trace(go.Scatter3d(x=[init[0]], y=[init[1]], z=[init[2]], mode='markers', marker=dict(color=colors[0]), name=agent_key))
            fig.add_trace(go.Scatter3d(x=[p[0] for p in path], y=[p[1] for p in path], z=[p[2] for p in path], mode='lines', line=dict(color=colors[1]),  showlegend=False))
            fig.add_trace(go.Scatter3d(x=[goal[0]], y=[goal[1]], z=[goal[2]], mode='markers', marker=dict(color=colors[2]), showlegend=False))

        return fig
    
    def default_prompt(self, feedback_plan = {}):
        agents = self.agents
        obstacles = self.obstacles
        grid_size = self.config['grid_size']
    
        system_prompt = '''Plan paths for agents to navigate a 3D grid to reach their respective goals and avoid collision.
You are given:
1) a list of obstacle coordinates (x, y, z): locations of the obstacle grid cells, agents must avoid them.
2) a list of [([name], [init], [goal]) tuples], [init] and [goal] are 3D coordinates of the initial position and goal position of agent named [name].
3) a previous plan, if any, and why it failed. Analyze this information and re-plan a collision-free path.

How to plan a <path>:
1) Make sure each path does not touch any obstacle or another agent.
2) Create a set of points for each agent to go from their init coordinates to their goal coordinates.
3) Make sure the coordinates are exactly one step away from each other, in one direction. Note - you may only move in one direction at a time.
Example of a <path>: [(1,1,1), (1,1,2), (1,1,3),...]

Output Instruction:
First output PLAN
Then, for each agent, output the coordinates of their path, each agent on a new line. 
EXAMPLE: 'PLAN
NAME Alice PATH[<path>]
NAME Bob PATH[<path>]
NAME Chad PATH[<path>]
NAME Dave PATH[<path>]'
        '''
        has_feedback = False if not feedback_plan else self.any_feedback(feedback_plan['feedback'])
    
        user_prompt_data = []
        obstacle_str = " ".join([f"({', '.join(map(str, obs['coords']))})" for obs in obstacles])
        agents_str = [f"Agent {agent} init: ({', '.join(map(str, agents[agent]['init']))}) goal: ({', '.join(map(str, agents[agent]['goal']))})" for agent in agents]
        user_info = f'''At the current step: Grid size: {' x '.join([str(grid_size)] * 3)}
Obstacles: {obstacle_str}
{agents_str[0]}
{agents_str[1]}
{agents_str[2]}
{agents_str[3]}'''
        user_prompt_data.append(user_info)
        
        
        if has_feedback:
            fback_join = [feedback_plan['plan'] + '.\n' if feedback_plan['plan'] else '', "\n".join([f for agent in feedback_plan['feedback'] for f in agent])]
            feedback = f'''
Feedback: the previous plan failed: {fback_join[0]}
Below are the specific reasons why it failed:
{fback_join[1]}
Use this information to try again, update this plan so it has complete, collision-free, strictly one-step-apart paths. '''
            user_prompt_data.append(feedback)
        
        user_prompt = '\n'.join(user_prompt_data + ['Your reasoning and plan is'])
    
        return system_prompt , user_prompt, feedback if has_feedback else ''
    
    def parse_response(self, response):
        '''
        inputs - env, response
        output - path, feedback : list[list[str]] - where the first list corresponds to the agents
        '''
        plan_ind = response.find('PLAN')
        if plan_ind == -1:
            return [], {'plan' : '', 'feedback' : [['No PLAN statement found']]}
        # If there's multiple PLANs, it goes error :(
        plan_ind = -1
        vals = response.split('\n')
        for i, l in enumerate(vals):
            if 'PLAN' in l:
                if vals[i+1].find('NAME') != -1:
                    plan_ind = i
                    break
        if plan_ind == -1: return [], {'plan' : '', 'feedback' : [['No parseable PLAN statement found']]}
            
        plan_vals = vals[i:i+5]
        info_extr = []
        feedback = []
        for action in plan_vals[1:]:
            extr = re.findall(r'NAME (\w+) PATH *(\[\((?:\d+, ?\d+, ?\d+)\)(?:, ?\((?:\d+, ?\d+, ?\d+)\))*\])', action)
            if not extr or len(extr[0]) != 2:
                info_extr.append({'':''})
                feedback.append([f'The following line could not be parsed: {action}'])
                continue
            feedback.append([''])
            info_extr.append({extr[0][0] : extr[0][1]})
            points = re.findall(r'\(\d+, ?\d+, ?\d+\)',extr[0][1])
            path = []
            for point in points:
                path.append([int(c.strip()) for c in point.strip('()').split(',')])
            info_extr[-1][extr[0][0]] = path
        pth = {list(l.keys())[0] : list(l.values())[0] for l in info_extr}
        return pth, {'plan' : '\n'.join(vals[i:i+5]), 'feedback' : feedback}
    
    def agent_plan_to_str(agent_plans):
        agents = []
        formatting = lambda x: ','.join([str(v) for v in x])
        for agent, path in agent_plans.items():
            agents.append(f'{agent}: [')
            agents[-1] += ','.join([f'({formatting(point)})'for point in path])
            agents[-1] += ']'
        return '\n'.join(agents)
    
    def tool_call_backend(self, agent_plans):
        # Checking for tool call errors

        feedback = [[] for x in self.agents]
        if self.agents != list(agent_plans.keys()):
            for i, agent in enumerate(self.agents):
                if agent not in list(agent_plans.keys()):
                    feedback[i].append(f'Agent {agent} not found in plan. Remember to include all paths for all agents in ONE tool call. ')
            
            if len(self.agents) < len(agent_plans.keys()):
                feedback[i].append('There were too many agents provided')
            
        for i, agent in enumerate(self.agents):
            if agent not in list(agent_plans.keys()) or not agent_plans[agent]:
                feedback[i].append(f'No path found for Agent {agent}')
        
        if self.any_feedback(feedback):
            return agent_plans, {'plan' : '', 'feedback' : feedback}, True
        
        paths = agent_plans
        paths_str = env.agent_plan_to_str(agent_plans)

        paths, feedback = self.validate_paths(paths, {'plan' : paths_str, 'feedback' : feedback})

        return paths, feedback, self.any_feedback(feedback['feedback'])


        


    
    
    def validate_paths(self, paths, feedback):
        if not feedback['plan']:
            return paths, feedback
        formatting = lambda x: ','.join([str(v) for v in x])
        obstacles = [tuple(val['coords']) for val in self.obstacles] + [v['goal']for v in self.agents.values()] + [v['init']for v in self.agents.values()]
        max_len = max([len(paths[agent]) for agent in paths])
        paths_agents = np.full((len(paths), max_len, 3), self.config['grid_size'] * 3, dtype=int)
        for i, agent in enumerate(paths):
            paths_agents[i, :len(paths[agent])] = paths[agent]

        path_agents = paths_agents.swapaxes(0,1)

        for k, agent_name in enumerate(paths):
            
            if feedback['feedback'][k]:
                continue
            path_feedback = []
            if sum(abs(np.array(paths[agent_name][-1]) - np.array(self.agents[agent_name]['goal']))) > 1:
                path_feedback.append(f'The path did not connect to the goal: POINT COMPARED - {formatting(paths[agent_name][-1])} | GOAL POINT - {formatting(self.agents[agent_name]["goal"])}.')
            if sum(abs(np.array(paths[agent_name][0]) - np.array(self.agents[agent_name]['init']))) > 1:
                path_feedback.append(f'The path did not connect to the init: POINT COMPARED - {formatting(paths[agent_name][0])} | INIT POINT - {formatting(self.agents[agent_name]["init"])}.')
            j = 1
            for l, point in enumerate(paths[agent_name]):

                if point in obstacles:
                    print('Collision detected')
                    path_feedback.append(f'Collision detected: {agent_name}: Path point ({formatting(point)}) collided with the following obstacle ({formatting(point)}).')
                if path_agents[l].tolist().count(path_agents[l][k].tolist()) > 1:
                    inds = [p if path_agents[l][p].tolist() == path_agents[l][k].tolist() and p != k else -1 for p, pont in enumerate(path_agents[l])]
                    while -1 in inds: inds.remove(-1)
                    o_agent_names = [name if r in inds else '' for r, name in enumerate(paths.keys())]
                    while '' in o_agent_names: o_agent_names.remove('')
                    o_points = [path_agents[l][q] for q in inds]
                    path_feedback.append(f"Collision detected: {agent_name}: Path point ({formatting(path_agents[l][k])}) collided with {', '.join([n + ' : (' + formatting(o_points[i]) + ')' for i, n in enumerate(o_agent_names)])}")
                if j == len(paths[agent_name]): continue
                next_point = paths[agent_name][j]
                diff = sum(abs(np.array(point) - np.array(next_point)))
                if diff == 0:
                    path_feedback.append(f'Duplicate points in one path: {agent_name}: Path point: ({formatting(point)}) is the same as ({formatting(next_point)}).')
                if diff > 1:
                    path_feedback.append(f'One or more of the coordinates in the following points were not exactly one unit apart: {agent_name}: Path points: [({formatting(point)}),({formatting(next_point)})]')
                j += 1
            feedback['feedback'][k] = path_feedback
        return paths, feedback
    
    def prompt(self, method, feedback_plan = {'feedback' : [[], []]}):
        methods = ['default']
        if method not in methods:
            raise ValueError(f'Prompting method not found. Must be one of: {methods}')
        if method.lower() == 'default':
            return self.default_prompt(feedback_plan)

    def any_feedback(self, feedback : list) -> bool:
        return max([any(v) for v in feedback])
        

    async def run(self, model : str, async_client : Union[openai.AsyncOpenAI], max_tries : int, prompting_method : str, save_folder : Path = Path()):
        '''
        Parameters
        ----------
        model : str
            OpenAI Model Identifier
        async_client : openai.AsyncOpenAI
            Asynchronus client
        max_tries : int
            Maximum number of replan tries
        prompting_method : str
            Must be a prompting method compatible with self.prompt
        save_folder : pathlib.Path
            Path to the folder used for save results for this run. Primarily used to run multiple envs in parallel
        '''
        config = {
            'service' : 'OpenAI',
            'model' : model,
            'max_tries' : max_tries,
            'prompting_method' : prompting_method,
            'success' : False,
            'tries' : 0,
        }
        log = []
        if save_folder == Path():
            path = (self.setup_data_path/'..' / '..' / 'logs' / self.config['edition'] / model / prompting_method).resolve()
            if not os.path.exists(path):
                os.makedirs(path)
            drs = os.listdir(path)
            path = path / f'run_{len(drs)}'
            os.makedirs(path)
            save_folder = path
        system_prompt, user_prompt = self.prompt(prompting_method)
        i = 0
        print(f"Querying_{i} in Env: {self.config['env_num']}")
        response = await async_client.chat.completions.create(messages=[
            {'role' : 'system', 'content' : system_prompt},
            {'role' : 'user', 'content' : user_prompt},
        ], model=model)
        i += 1
        log.append([{'role' : 'system', 'content' : system_prompt}, {'role' : 'user', 'content' : user_prompt}, {'role' : 'assistant', 'content' : response.choices[0].message.content}])
        print('===== RESPONSE ======')
        print(f"Env: {self.config['env_num']}")
        print(response.choices[0].message.content)
        paths, feedback = self.parse_response(response.choices[0].message.content)
        paths, feedback = self.validate_paths(paths, feedback)
        print('===== FEEDBACK =====')
        print(f"Env: {self.config['env_num']}")
        print(feedback)
        config['tries'] = i
        if not self.any_feedback(feedback['feedback']):
            config['success'] = True

        with open(save_folder / f'env_{self.config["env_num"]}.json', 'w') as file:
            json.dump({'log' : log, 'config' : config}, file, indent=4)
        

        while self.any_feedback(feedback['feedback']) and i < max_tries:
            print(f"Prompting_{i} in Env: {self.config['env_num']}")
            system_prompt, user_prompt = self.prompt(prompting_method, feedback)
            
            print(f"Querying_{i} in Env: {self.config['env_num']}")
            response = await async_client.chat.completions.create(messages=[
            {'role' : 'system', 'content' : system_prompt},
            {'role' : 'user', 'content' : user_prompt},], model=model)
            i += 1
            log.append([{'role' : 'system', 'content' : system_prompt}, {'role' : 'user', 'content' : user_prompt}, {'role' : 'assistant', 'content' : response.choices[0].message.content}])
            print('===== RESPONSE ======')
            print(f"Env: {self.config['env_num']}")
            print(response.choices[0].message.content)
            paths, feedback = self.parse_response(response.choices[0].message.content)
            paths, feedback = self.validate_paths(paths, feedback)
            print('===== FEEDBACK =====')
            print(f"Env: {self.config['env_num']}")
            print(feedback)
            config['tries'] = i
            if not self.any_feedback(feedback['feedback']):
                config['success'] = True
            with open(save_folder / f'env_{self.config["env_num"]}.json', 'w') as file:
                json.dump({'log' : log, 'config' : config}, file, indent=4)
        
        

        self.paths[model] = paths




# Util functions

def cubes(size, pos_x, pos_y, pos_z, color):
    # create points
    x, y, z = np.meshgrid(
        np.linspace(pos_x-size/2, pos_x+size/2, 2), 
        np.linspace(pos_y-size/2, pos_y+size/2, 2), 
        np.linspace(pos_z-size/2, pos_z+size/2, 2),
    )
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    return go.Mesh3d(x=x, y=y, z=z, alphahull=1, flatshading=True, color=color, lighting={'diffuse': 0.1, 'specular': 2.0, 'roughness': 0.5})

def clamp_color(val):
    if val < 0:
        val = 0
    if val > 255:
        val = 255
    return val

def clamp_op(val):
    if val < 0:
        val = 0
    if val > 1:
        val = 1
    return val