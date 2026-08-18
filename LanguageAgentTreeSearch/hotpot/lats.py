import itertools
import numpy as np
from functools import partial
from models import gpt
import wikienv, wrappers
import requests
import logging
import random
import concurrent.futures
import time
import os

API_BASE = os.getenv("CUSTOM_API_BASE", "http://localhost:8000")

def notify_program_completion(program_id, idx, status, time_info=None):
    payload = {
        "program_id": program_id,
        "task_idx": idx,
        "status": status,
        "time_info": time_info
    }
    try:
        url = f"{API_BASE}/program/complete"
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        logging.info(f"[{program_id}] Sinal de término enviado com sucesso: {response.status_code}")
    except Exception as e:
        logging.error(f"[{program_id}] Falha ao enviar sinal de término: {e}")

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

def get_value(task, x, y, n_evaluate_sample, failed_trajectories, reflection_map, cache_value=True, program_id="default_prog"):
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y, unique_trajectories, reflection_map)
    logging.info(f"[{program_id}] Current: {x}")
    logging.info(f"[{program_id}] Current: {y}")
    
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
        
    logging.info(f"[{program_id}] VALUE PROMPT: {value_prompt}")
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None, program_id=program_id)
    logging.info(f"[{program_id}] VALUE OUTPUTS: {value_outputs}")
    
    value = task.value_outputs_unwrap(value_outputs)
    logging.info(f"[{program_id}] VALUES: {value}")
    
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, failed_trajectories, reflection_map, cache_value=True, program_id="default_prog", step_stats=None, phase="expansion"):
    values = []
    local_value_cache = {}
    
    # Identifica os 'ys' únicos para evitar chamadas de API redundantes
    unique_ys = []
    for y in ys:
        if y not in unique_ys:
            unique_ys.append(y)
    if step_stats is not None:
        if phase == "expansion":
            step_stats["start_evaluation_requests"] += len(unique_ys) * n_evaluate_sample
        elif phase == "rollout":
            step_stats["rollout_evaluation_requests"] += len(unique_ys) * n_evaluate_sample

    # Função auxiliar para enviar ao ThreadPool
    def fetch_value(y):
        val = get_value(task, x, y, n_evaluate_sample, failed_trajectories, reflection_map, cache_value=cache_value, program_id=program_id)
        return y, val

    # Executa as avaliações de 'get_value' em paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_value, unique_ys)
        
        # Mapeia os resultados de volta para o cache local
        for y, val in results:
            local_value_cache[y] = val

    # Monta a lista final de 'values' preservando a ordem original do array 'ys'
    for y in ys:
        values.append(local_value_cache[y])
        
    return values

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, failed_trajectories, reflection_map, program_id="default_prog", step_stats=None, phase="expansion"):
    if step_stats is not None:
        if phase == "expansion":
            step_stats["start_expansion_requests"] += n_generate_sample
        elif phase == "rollout":
            step_stats["rollout_expansion_requests"] += n_generate_sample   
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
        # print(f"[{program_id}] 6) REFLECTION: Gerando reflexão sobre trajetória falha...")
        # print(f"[{program_id}] generating reflections")
        reflection_map.extend(task.generate_self_reflection(unique_trajectories, x, program_id=program_id))
        
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
        
    logging.info(f"[{program_id}] PROMPT: {prompt}")
    samples = gpt(prompt, n=n_generate_sample, stop=stop, program_id=program_id)
    return [y + _ for _ in samples]

def get_unique_trajectories(failed_trajectories, num=5):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories

class Node:
    def __init__(self, state, question, parent=None):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }
    
def node_trajectory_to_text(node_string):
    lines = node_string.split('\n')
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            thought = line.split(", thought=")[1].split(", action=")[0].strip()
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue
        
        if depth != 0:
            if thought:
                formatted_lines.append(f"Thought {depth}: {thought}")
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    
    return '\n'.join(formatted_lines)

def collect_all_nodes(node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))
        return nodes

def collect_trajectory(node):
    trajectory = []
    while node:
        trajectory.append(str(node))
        node = node.parent
    return '\n'.join(reversed(trajectory))

def lats_search(args, task, idx, iterations=30, to_print=True, program_id="default_prog"):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    
    # Isolate environment instance per thread/program
    env = wikienv.WikiEnv()
    env = wrappers.HotPotQAWrapper(env, split="train")
    env = wrappers.LoggingWrapper(env)
    
    def get_search_stats(current_env):
        # Desempacota as camadas do ambiente
        original_env = current_env
        while hasattr(current_env, 'env'):
            current_env = current_env.env
            
        time_info = current_env.get_time_info()
        time_info["cache_hits"] = time_info.get("cache_hits", None)
        time_info["cache_misses"] = time_info.get("cache_misses", None)
        
        # Resgata o tempo real de caminho crítico que injetamos (ou 0 se nenhuma busca foi feita)
        time_info["lats_critical_path_time"] = getattr(original_env, 'lats_real_time', 0.0)
        
        return time_info
    
    x = env.reset(idx=idx)
    if to_print:
        print(f"[{program_id}] {idx}", x)
        
    root = Node(state=None, question=x)
    all_nodes = []
    
    # Isolate global states into local variables
    failed_trajectories = []
    terminal_nodes = []
    reflection_map = []
    
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    step_stats = {
        "selection_time_s": 0.0,
        "expansion_time_s": 0.0,
        "evaluation_time_s": 0.0,
        "rollout_time_s": 0.0,
        "backprop_time_s": 0.0,
        "start_expansion_requests": 0,    # NOVO
        "start_evaluation_requests": 0,   # NOVO
        "rollout_expansion_requests": 0,  # NOVO
        "rollout_evaluation_requests": 0  # NOVO
    }

    iterations_used = 0

    for i in range(iterations):
        if getattr(args, 'kill_event', None) and args.kill_event.is_set():
            logging.info(f"[{program_id}] Kill event detected! Stopping iterations.")
            notify_program_completion(program_id, idx, status="interrupted")
            raise InterruptedError("Hard stopped by orchestrator timeout")
        logging.info(f"[{program_id}] Iteration {i + 1}...")
        iterations_used = i + 1
        
        # Medindo Selection
        t0 = time.time()
        node = select_node(root, program_id)
        
        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"[{program_id}] Need to backtrack...")
            node = select_node(root, program_id)
        step_stats["selection_time_s"] += time.time() - t0
        
        if node is None:
            break

        if node.is_terminal and node.reward == 1:
            time_info = get_search_stats(env)
            time_info.update(step_stats) # Adiciona as métricas ao retorno
            time_info["iterations_used"] = iterations_used
            notify_program_completion(program_id, idx, status="success_selection", time_info=time_info)
            return node.state, node.value, all_nodes if all_nodes else [node], node.reward, node.em, time_info
        
        # Medindo Expansion
        t0 = time.time()
        expand_node(node, args, task, env, failed_trajectories, reflection_map, program_id, step_stats)
        
        while node.is_terminal or not node.children:
            # Caso caia no limite de profundidade e re-selecione, o tempo conta para expansão
            node = select_node(root, program_id)
            expand_node(node, args, task, env, failed_trajectories, reflection_map, program_id, step_stats)
        step_stats["expansion_time_s"] += time.time() - t0

        # Medindo Evaluation
        t0 = time.time()
        value = evaluate_node(node, args, task, failed_trajectories, reflection_map, program_id, step_stats)
        step_stats["evaluation_time_s"] += time.time() - t0
        
        # Medindo Rollout
        t0 = time.time()
        reward, terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, idx, env, failed_trajectories, reflection_map, max_depth=4, program_id=program_id, step_stats=step_stats)
        terminal_nodes.append(terminal_node)
        step_stats["rollout_time_s"] += time.time() - t0

        if terminal_node.reward == 1:
            time_info = get_search_stats(env)
            time_info.update(step_stats)
            time_info["iterations_used"] = iterations_used
            notify_program_completion(program_id, idx, status="success_rollout", time_info=time_info)
            return terminal_node.state, terminal_node.value, [terminal_node], terminal_node.reward, terminal_node.em, time_info

        # Medindo Backpropagation
        t0 = time.time()
        backpropagate(terminal_node, reward, program_id)
        step_stats["backprop_time_s"] += time.time() - t0
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        # Check for terminal nodes with a reward of 1
        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logging.info(f"[{program_id}] Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            time_info = get_search_stats(env)
            time_info["iterations_used"] = iterations_used
            notify_program_completion(program_id, idx, status="success_tree_search", time_info=time_info)
            return best_node.state, best_node.value, all_nodes if all_nodes else [best_node], best_node.reward, best_node.em, time_info
    
        for j, (n, v) in enumerate(all_nodes):
            logging.info(f"[{program_id}] Node {j+1}: {str(n)}")

        logging.info(f"[{program_id}] State of all_nodes after iteration {i + 1}: {all_nodes}")

    all_nodes_list = collect_all_nodes(root)
    all_nodes_list.extend(terminal_nodes)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    
    if best_child.reward == 1:
        logging.info(f"[{program_id}] Successful trajectory found")
    else:
        logging.info(f"[{program_id}] Unsuccessful trajectory found")
        
    if best_child is None:
        best_child = root
    time_info = get_search_stats(env)
    time_info.update(step_stats)
    time_info["iterations_used"] = iterations_used
    status_final = "finished_success" if best_child and best_child.reward == 1 else "finished_max_iterations"
    notify_program_completion(program_id, idx, status=status_final, time_info=time_info)
    return best_child.state, best_child.value, all_nodes_list, best_child.reward, best_child.em, time_info

def select_node(node, program_id="default_prog"):
    # print(f"[{program_id}] 1) SELECTION: Selecionando nó na profundidade {node.depth}")
    while node and node.children:
        logging.info(f"[{program_id}] Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        
        if len(terminal_children) == len(node.children):
            logging.info(f"[{program_id}] All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"[{program_id}] Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1
        
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logging.info(f"[{program_id}] Selected node at depth {node.depth} with UCT {node.uct()}.")

        
    return node  # This will return None if all paths from the root are exhausted

def expand_node(node, args, task, env, failed_trajectories, reflection_map, program_id="default_prog", step_stats=None):
    # print("LIMITE PROFUNDIADE:",args.depth_limit)
    if node.depth >= args.depth_limit:
        logging.info(f"[{program_id}] Depth limit reached")
        print(f"[{program_id}] Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample, env, failed_trajectories, reflection_map, program_id, step_stats=step_stats, phase="expansion")
    node.children.extend(new_nodes)

def rollout(node, args, task, idx, env, failed_trajectories, reflection_map, max_depth=4, program_id="default_prog", step_stats=None):
    logging.info(f"[{program_id}] ROLLING OUT")
    depth = node.depth
    # print(f"[{program_id}] 4) SIMULATION: Fazendo rollout a partir da profundidade {depth}...")

    n = args.n_rollout
    rewards = [0]
    while not node.is_terminal and depth < max_depth:
        if getattr(args, 'kill_event', None) and args.kill_event.is_set():
            logging.info(f"[{program_id}] Kill event detected! Stopping rollout.")
            raise InterruptedError("Hard stopped by orchestrator timeout")
        logging.info(f"[{program_id}] ROLLING OUT {depth}")
        new_states = []
        values = []
        while len(new_states) == 0:
            # Usa phase="rollout"
            new_states = generate_new_states(node, args, task, n, env, failed_trajectories, reflection_map, program_id, step_stats=step_stats, phase="rollout")

        for state in new_states:
            if state.is_terminal:
                return state.reward, state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        while len(values) == 0:
            # Usa phase="rollout"
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample, failed_trajectories, reflection_map, program_id=program_id, step_stats=step_stats, phase="rollout")
            
        max_value_index = values.index(max(values))
        rewards.append(max(values))
        node = new_states[max_value_index] 
        depth += 1
        if depth == max_depth:
            rewards = [-1]
    
    logging.info(f"[{program_id}] ROLLOUT FINISHED")
    return sum(rewards) / len(rewards), node

def generate_new_states(node, args, task, n, env, failed_trajectories, reflection_map, program_id="default_prog", step_stats=None, phase="expansion"):
    # print(f"[{program_id}] 2) EXPANSION: Gerando {n} novas ações...")
    prompt = generate_prompt(node)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample, stop="Observation", failed_trajectories=failed_trajectories, reflection_map=reflection_map, program_id=program_id, step_stats=step_stats, phase=phase)
    logging.info(f"[{program_id}] SAMPLED ACTION: {sampled_actions}")
    
    # print(f"\n[{program_id}] TEXTO GERADO PELO LLAMA 3.1:\n{sampled_actions}\n")
    tried_actions = []
    
    unique_states = {}  # Store unique states here
    for action in sampled_actions:
        new_state = node.state.copy()  # Make a copy of the parent node's state

        thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue  # Skip if this state already exists

        tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            inicio_step = time.time()
            
            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")
            
            tempo_step = time.time() - inicio_step
            if not hasattr(env, 'lats_real_time'):
                env.lats_real_time = 0.0
            env.lats_real_time += tempo_step
            # Update the new state dictionary
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = obs

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1
            if r == 1:
                new_node.em = info.get('em')
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"[{program_id}] NEW NODE: {new_node}")
            logging.info(f"[{program_id}] Feedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory(new_node)
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

    return list(unique_states.values())  # Return unique nodes as a list


def evaluate_node(node, args, task, failed_trajectories, reflection_map, program_id="default_prog", step_stats=None):
    # print(f"[{program_id}] 3) EVALUATION: Avaliando {len(node.children)} nós filhos...")
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample, failed_trajectories, reflection_map, program_id=program_id, step_stats=step_stats, phase="expansion")    
    logging.info(f"[{program_id}] Length of votes: {len(votes)}")
    logging.info(f"[{program_id}] Length of node.children: {len(node.children)}")
    
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    for i, child in enumerate(node.children):
        child.value = votes[i] 
    
    return sum(votes) / len(votes) if votes else 0


def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)

def backpropagate(node, value, program_id="default_prog"):
    # print(f"[{program_id}] 5) BACKPROPAGATION: Subindo valor {value:.2f} pela árvore...")
    while node:
        node.visits += 1
        if node.is_terminal:
            if node.reward == 0:
                node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                logging.info(f"[{program_id}] Backpropagating with reward 0 at depth {node.depth}. New value: {node.value}.")
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logging.info(f"[{program_id}] Backpropagating with reward 1 at depth {node.depth}. New value: {node.value}.")
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logging.info(f"[{program_id}] Backpropagating at depth {node.depth}. New value: {node.value}.")

        node = node.parent

def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))