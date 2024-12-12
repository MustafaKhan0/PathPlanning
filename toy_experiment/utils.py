import random
from pathlib import Path
import os
import json
import pprint
from sys import maxsize
from env import env
from openai import AsyncOpenAI
from typing import Union, List
import asyncio


def generate_new_env(setup_data_path : Path, count = 30, grid_size = 10, num_obstacles = 10, obstacle_size = 1, agent_names = ['Alice', 'Bob', 'Chad', 'Dave'],seed = random.randrange(maxsize), edition : str = ''):
    random.seed(seed)
    random_coords = lambda: [random.randint(1,grid_size), random.randint(1,grid_size), random.randint(1,grid_size)]
    if not edition:
        dirs = os.listdir(setup_data_path)
        edition =  f'Version_{len(dirs)}'
    dir_path = setup_data_path / edition
    os.mkdir(dir_path)
    

    config = {
        'grid_size' : grid_size,
        'num_obstacles' : num_obstacles,
        'obstacle_size' : obstacle_size,
        'agent_names' : agent_names,
        'seed' : seed, 
        'count' : count,
        'edition' : edition,
    }
    with open(dir_path / 'config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)
    
    
    for i in range(count):
        coords = []
        obstacles = []
        agents = {}
        for j in range(num_obstacles):
            obs_coord = random_coords()
            while obs_coord in coords:
                obs_coord = random_coords()
            coords.append(obs_coord)
            obstacles.append({'size' : obstacle_size, 'coords' : obs_coord})

        for name in agent_names:
            coord1 = random_coords()
            coord2 = random_coords()
            while coord1 in coords:
                coord1 = random_coords()
            coords.append(coord1)
            while coord2 in coords:
                coord2 = random_coords()
            coords.append(coord2)
            agents[name] = {'init' : coord2, 'path' : [], 'goal' : coord1}
        config['env_num'] = i
        with open(dir_path / f'env_{i}.json', 'w') as env_file:
            data_str = pprint.pformat({'config' : config, 'obstacles' : obstacles, 'agents' : agents}, compact=True, indent=4).replace("'",'"')
            env_file.write(data_str)
            
    
    return dir_path

def load_envs(setup_path : Path, edition : str):
    dir_path = setup_path / edition
    files = os.listdir(dir_path)
    

    files.remove('config.json')
    envs = []
    for file_name in list(sorted(files)):
        envs.append(env.from_json(dir_path / file_name))

    return envs

async def run_envs(envs : List[env], model : str, async_client : Union[AsyncOpenAI], max_tries : int, prompting_method : str, custom_path : Path = Path()):
    '''
    Parameters
    ----------
    envs : list[env]
        List of environments to run
    model : str
        OpenAI Model Identifier
    async_client : openai.AsyncOpenAI
        Asynchronus client
    max_tries : int
        Maximum number of replan tries
    prompting_method : str
        Must be a prompting method compatible with self.prompt
    custom_path : pathlib.Path
        Path to the folder used for save results for this run. Not recommended, but possible. 
    '''
    if custom_path == Path():
        path = (envs[0].setup_data_path/'..' / '..' / 'logs' / envs[0].config['edition'] / model / prompting_method).resolve()
        if not os.path.exists(path):
            os.makedirs(path)
        drs = os.listdir(path)
        path = path / f'run_{len(drs)}'
        os.makedirs(path)
        filepath = path
    else:
        filepath = custom_path
    tasks = []
    for env in envs:
        tasks.append(asyncio.create_task(env.run(model, async_client, max_tries, prompting_method, filepath)))
    print([t.done() for t in tasks])
    print(min(tasks, key=lambda x: x.done()))
    while not min(tasks, key=lambda x: x.done()):
        for i, task in enumerate(tasks):
            if task.done():
                print(f'Env_{i} has completed')
    print('All envs complete!')
    for i, task in enumerate(tasks):
        try:
            await task
        except Exception as e:
            print('===== EXCEPTION =====')
            print(f'Env: {i}')
            print(e)
    
    # Old solution for jupyter notebooks
    # loop = asyncio.get_event_loop()
    # big_task = loop.create_task()
