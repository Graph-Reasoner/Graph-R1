import json
import math
import re
from typing import Dict, Any
import networkx as nx  # 添加 networkx 导入
# from .tasks.Connected import Connected_Task  # 修改为相对导入
from .tasks.MCS import MCS_Task
from .tasks.TSP import TSP_Task
from .tasks.GED import GED_Task
from .tasks.MCP import MCP_Task
from .tasks.MVC import MVC_Task
from .tasks.MIS import MIS_Task
from .tasks.Connected import Connected_Task
from .tasks.Diameter import Diameter_Task
from .tasks.Neighbor import Neighbor_Task
from .tasks.Distance import Distance_Task

# def build_networkx_graph(graph_data):
#     """Convert JSON graph data to NetworkX graph."""
#     if isinstance(graph_data, str):
#         try:
#             graph_data = json.loads(graph_data) # 从json字符串转成python对象
#         except:
#             # print("Failed to parse graph JSON")
#             return None

#     try:
#         if not isinstance(graph_data, list):
#             G = nx.Graph()
            
#             # 添加节点和属性
#             for node in graph_data.get("nodes", []):
#                 G.add_node(node["id"], name=node["name"])
            
#             # 添加边
#             for link in graph_data.get("links", []):
#                 G.add_edge(link["source"], link["target"])
                
#             # print(f"Successfully built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
#             return G
#         else: # 处理graph_data包含多个图的情况（递归）
#             return [build_networkx_graph(g) for g in graph_data]        
#     except Exception as e:
#         # print(f"Failed to build graph: {e}")
#         return None

def build_networkx_graph(graph_data):
    """转换JSON图数据为NetworkX图。"""
    if isinstance(graph_data, str):
        try:
            graph_data = json.loads(graph_data)  # 从json字符串转成python对象
        except:
            # print("Failed to parse graph JSON")
            return None

    try:
        if not isinstance(graph_data, list):
            G = nx.Graph()
            
            # 添加节点和属性
            for node in graph_data.get("nodes", []):
                attrs = {}
                # 处理节点可能有label或name属性的情况
                if "label" in node:
                    attrs["label"] = node["label"]
                if "name" in node:
                    attrs["name"] = node["name"]
                G.add_node(node["id"], **attrs)
            
            # 添加边
            for link in graph_data.get("links", []):
                attrs = {}
                # 处理边可能有label或relation属性的情况
                if "label" in link:
                    attrs["label"] = link["label"]
                if "relation" in link:
                    attrs["relation"] = link["relation"]
                G.add_edge(link["source"], link["target"], **attrs)
            
            # print(f"Successfully built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
        else:  # 处理graph_data包含多个图的情况（递归）
            return [build_networkx_graph(g) for g in graph_data]        
    except Exception as e:
        # print(f"Failed to build graph: {e}")
        return None
    
# def connected_reward(completions, extra_info): # 此处有改动，把kwargs
def graph_reward(completions, extra_info, task_name):
    """Reward function for connected components task."""
    content = completions
    example = extra_info # 包含graph和exact_answer，不知道这个格式有没有问题
    # rewards = []
    reward = 0.0
    print("---GRAPH REWARD DEBUG---")
    
    # Get task instance
    # task_instance = Connected_Task()
    task_instance = globals()[f"{task_name}_Task"]()

    graph_json = example.get("graph", "{}")
    # print(f"graph_json: {graph_json}")
    
    graph = None
    # 将JSON转换为NetworkX图
    try:
        graph = build_networkx_graph(graph_json) # 在这里面处理graph_json转成python之后是个list的情况
    except Exception as e:
        # print(f"Error in build_networkx_graph: {e}")
        graph = None
    
    # 从extra_info中获取problem_text
    problem_text = example.get("problem_text", "")

    # 把图任务按照reward计算方式分类
    small_better = ["Distance", "MVC", "GED", "TSP"]
    big_better = ["Neighbor", "Connected", "Diameter", "MCP", "MIS", "MCS"]
    
    # Validate graph object
    res = 0.0
    if graph is None:
        reward = 0.0
        print("graph is None!")
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        return reward # 直接返回
    else:
        # Get result from task's check_solution with explicit graph
        try:
            # print(f"graph is {graph}")
            res = task_instance.check_solution(
                problem_id=None,
                response=content,
                graph=graph,
                problem_text=problem_text
            )
            print("Current result:", res)
        except Exception as e:
            print(f"Error in check_solution: {e}")
            reward = 0.0
            print("Reward:", reward)
            print("---END GRAPH REWARD---\n")
            return reward # 直接返回
                
            # rewards.append(reward)

    # 提取精确解：
    expected = example.get("exact_answer", 0)

    # 把ground_truth从str转换为int
    if expected: # expected is not null
        expected = int(float(expected))
        print(f"Expected answer: {expected}")
        # 计算奖励:
        # - 完全正确给 2.0 分
        # - 找到部分正确答案给 (res/expected)* 1.0 分【但似乎应该只适用于找最大的问题，如果是最短路是不是应该反着来】
        # - 无效结果给 0 分
        if res == expected:
            reward = 2.0
        elif res > 0: # 有效答案
            # if task_name == "TSP":
            #     if res < expected:
            #         reward = 0.5 * (res / expected) ** 2
            #     else:
            #         reward = 0.5 * (expected / res) ** 2
            # else:
            #     if res < expected:
            #         reward = (res / expected) * 0.5
            #     else:
            #         reward = (expected / res) * 0.5
            if task_name in small_better:
                if task_name != "TSP":
                    reward = (expected / res) * 0.5
                else:
                    reward = 0.5 * ((expected / res) ** 2) # TSP问题，用一个平方项让奖励梯度更陡峭。
            else:
                reward = (res / expected) * 0.5
        else: # 无效答案
            # reward = 0.0
            reward = -1.0 # 增大无效答案的惩罚
        print("Reward:", reward)
        print("---END GRAPH REWARD---\n")
        # return rewards
        return reward
    else:
        # GED_hard和MCS_hard的expected=null，怎么办呢？
        # 这种情况也只在test的时候会出现，所以直接返回res就行，不用担心reward混乱的问题
        print(f"No exact answer, return res only: {res}")
        print("---END GRAPH REWARD---\n")
        return res


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    # pattern = r"^.*?<think>\n.*?</think>\n<answer>.*?</answer>.*?$" # 比较宽松的格式，对think前和answer后没有要求
    # pattern = r"^<think>\n.*?</think>.*?$" # 只要求think，不要求answer
    pattern = r"^<think>\n.*?$" # 更宽松版本的thinkonly，也正好跟inference时设置的限制是一致的
    # 严格的格式，先think再answer，answer完就结束，answer中间需要有真正的答案。
    # pattern = r"^<think>\n.*?</think>\n<answer>.*?\[\s*([A-Z\s,]*)\s*\].*?</answer><|im_end|>$"
    content = completions
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)

    # # print("正则表达式匹配结果:", "成功" if match else "失败")
    # if match:
    #    # print("匹配到的城市列表:", match.group(1))
    # return [1.0] if match else [0.0]
    return 1.0 if match else 0.0
    # return 2.0 if match else 0.0


def tag_count_reward(completions, **kwargs):
    """Reward function that checks if we produce the desired number of think and answer tags."""
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count
    content = completions
    return count_tags(content)

from string_repetition import StringRepetitionDetector

def detect_repeat(solution_str: str):
    detector = StringRepetitionDetector(
        min_length=20,     # 最小重复长度
        min_repeats=5      # 最小重复次数
    )
    result = detector.detect_string(solution_str, parallel=True)
    result = result.has_repetition # 找到重复模式即为True，否则为False
    return result

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 extra_info: Dict[str, Any] = None,
                 task_name: str = "Connected"
                ) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    # print(" Processing New Sample ".center(80, '='))
    print(f"\n[Problem Text]\n{extra_info.get('problem_text')}")
    print(f"\n[Model Response]\n{solution_str}")
    print(f"\n[Ground Truth]\n{extra_info.get('exact_answer')}")
    # print(f"\n[Extra Info]\n{extra_info.get('graph')}")
    

    # Validate response structure
    format_score = format_reward(solution_str) 
    # format_correct = True if format_score == 1.0 else False
    format_correct = True if format_score == 1.0 else False

    if format_correct:
        format_score_scaled = 1
    else:
        format_score_scaled = 0
    
    # print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    # print(f"  Format score: {format_score}")
    
    # 为了更好地激励格式，格式分由两部分构成，一部分是绝对格式分，一部分是贪心格式分
    # 训练后期，不需要tag_count_reward，只需要绝对格式分，避免模型为了奖励而hack格式
    # # print(f"absolute format score: {format_score}")
    # tag_count_reward_score = tag_count_reward(solution_str)
    # # print(f"tag count reward: {tag_count_reward_score}")
    # format_score = 0.5 * format_score + 0.5 * tag_count_reward_score

    # answer_score = connected_reward(solution_str, extra_info)

    # task_name_extract = task_name.split("_")[0]
    answer_score = graph_reward(solution_str, extra_info, task_name)
    if answer_score == 2.0:
        answer_score_scaled = 1
    else:
        answer_score_scaled = 0

    repeat_appear = detect_repeat(solution_str)

    if repeat_appear:
        repeat_score = -1.0 # 这个值有待商榷
        repeat_score_scaled = 1.0 # 1表示出现，0表示没出现
    else:
        repeat_score = 0.0    
        repeat_score_scaled = 0.0
    
    # total_score = format_score + answer_score
    total_score = format_score + answer_score + repeat_score
    # print("\n" + "-"*80)
    # print(f" Final Score ".center(80, '-'))
    # print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Repeat: {repeat_appear}")
    # print(f"  Total: {total_score}")
    print("="*80 + "\n")
    
    # output = {
    #     "score": total_score,
    #     "extra_info": {
    #         "outcome_score": answer_score_scaled, # scaled:答案完全正确给1，错误给0
    #         "format_score": format_score_scaled # scaled:格式正确给1，错误给0
    #     }
    # }
    output = {
        "score": total_score,
        "outcome_score": answer_score_scaled, # scaled:答案完全正确给1，错误给0
        "format_score": format_score_scaled, # scaled:格式正确给1，错误给0
        "repeat_score": repeat_score_scaled
    }

    return output
