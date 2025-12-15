import json
import os
import random
import re
from Levenshtein import ratio
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def process_expression(s):
    # 使用正则表达式移除所有运算符（=+、-、*、/）周围的空格
    return re.sub(r'\s*([=+\-*/])\s*', r'\1', s)

def basic_verify(content, sol):
    '''
        Args:
            content: 模型生成的回答
            sol: 为ground truth
        Returns:
            reward: 1.0表示正确, 0.0表示错误
        Warning: basic_verify 无法识别 \boxed{40} 当中的 40
        >>> content = '\boxed{40}'
        >>> sol = '40'
        >>> basic_verify(content, sol)
        >>> # Output: 0.0
    '''
    answer = parse(content)
    verified_res = float(verify(answer, parse(sol)))
    # print('check verified res: ', verified_res)
    if verified_res > 0:
        reward = 1.0
    else:
        reward = 0.0
    return reward



def parse_latex_gt(sol):
    '''Args:
            sol: ground truth, 例如 '40' 或者 '$40$'
        Returns:
            gold_parsed: 解析后的ground truth
        >>> content = '$40$'
        >>> parse_latex_gt(content)
        >>> [40, '40']
    '''
    gold_parsed = parse(
        sol,
        extraction_mode="first_match", # 提取模式：只提取第一个匹配的数学表达式
        extraction_config=[LatexExtractionConfig()], # 使用LaTeX提取配置
    )
    return gold_parsed


def parse_latex_response(content):
    '''
        nits=False:
            作用：不自动修正LaTeX中的小错误
            例子：不会自动修正 x ^ 2 为 x^2，或 \frac { 1 } { 2 } 为 \frac{1}{2}
            用途：保持学生输入的原始格式，避免过度修正影响评分准确性
        
        malformed_operators=False
            作用：不修复错误的运算符格式
            例子：不会自动修正 x * * 2 为 x^2，或 a + + b 为 a + b
            用途：如果学生写错了运算符，系统不会自动"帮忙"修正，确保评分的严格性
        
        basic_latex=True
            作用：启用基础LaTeX命令解析
            例子：能识别 \frac{1}{2}, \sqrt{x}, x^2, x_1 等常见LaTeX语法
            用途：处理标准的数学表达式格式
        
        equations=True
            作用：启用方程式处理
            例子：能处理 x + 2 = 5, y = 2x + 1 这样的等式
            用途：用于需要验证方程解答的题目

        boxed=True
            作用：识别并处理 \boxed{} 包围的内容
            例子：从 计算过程... \boxed{x = 3} 中提取最终答案
            用途：学生通常用 \boxed{} 标记最终答案
        
        units=True
            作用：处理带单位的数值
            例子：能识别 5 \text{ cm}, 3.14 \text{ m/s} 等
            用途：物理、化学等需要单位的题目
        
        boxed_match_priority=0
            作用：当文本中有多个可能的答案时，优先提取 \boxed{} 中的内容
            例子：在 中间计算: 2x = 6, 所以 \boxed{x = 3} 中，优先提取 x = 3 而不是 2x = 6
            用途：确保提取到学生认为是最终答案的部分
        
        
        try_extract_without_anchor=False
            作用：不尝试从没有明确标记的文本中提取数学表达式
            例子：如果文本是 答案是三 而不是 答案是 3，不会尝试提取
            用途：避免误提取，确保只处理明确的数学表达式
    '''
    latex_extraction_config = LatexExtractionConfig(
        normalization_config=NormalizationConfig(
                nits=False, # 不进行细节修正（如空格、格式等小问题）
                malformed_operators=False, # 不修复格式错误的运算符
                basic_latex=True, # 启用基础LaTeX语法处理
                equations=True, # 启用方程式处理
                boxed=True, # 启用对 \boxed{} 命令的处理（通常用于标记最终答案）
                units=True, # 启用单位处理
            ),
        # Ensures that boxed is tried first
        boxed_match_priority=0, #  设置 \boxed{} 的匹配优先级为最高（0是最高优先级）
        try_extract_without_anchor=False, # 不尝试在没有锚点的情况下提取
    )
    answer_parsed = parse(
                content,
                extraction_mode="first_match",
                extraction_config = [latex_extraction_config],
    )

    return answer_parsed



def second_verify(content, sol):
    gold_parsed = parse_latex_gt(sol)
    # 若成功精炼出 ground truth
    if len(gold_parsed) != 0:
        # 尝试从 response content 中提取
        answer_parsed = parse_latex_response(content)
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
       
        # 
        threshold = float(verify(answer_parsed, gold_parsed))
        if threshold > 0:
            reward = 1.0
        else:
            reward = 0.0
    return reward


def extract_response_content_from_answer_tag(content):
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if content_matches:
        student_answer = content_matches[-1].strip() 
    else:
        student_answer=content
    return student_answer




def extract_ans_verify(content, sol):
    # 若rewards还为0
    # 提取 content 中 <answer> 和 </answer> 之间的内容
    # content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    # student_answer = content_matches[-1].strip() if content_matches else content.strip()
    student_answer = content

    # 使用正则表达式在字符串 sol 中搜索 <answer>...</answer> 中间的内容
    sol_match = re.search(r'<answer>(.*?)</answer>', sol) # 如果找到匹配，返回 Match 对象；否则返回 None
    # sol_match.group(1) - 提取第一个捕获组的内容（即 <answer> 和 </answer> 之间的内容）.strip() - 去除首尾空白字符
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    if ground_truth[0] == "$" and ground_truth[-1] == "$":
        ground_truth = ground_truth[1:-1]
    if student_answer[0] == "$" and student_answer[-1] == "$":
        student_answer = student_answer.replace("$","")
    student_answer = process_expression(student_answer)
    ground_truth = process_expression(ground_truth)
    if student_answer == ground_truth:
        reward = 1.0
    else:
        reward = 0.0
    return reward


def default_accuracy_reward(content, sol):
    '''
        content:  模型生成的回答
        sol: 为ground truth
    '''
    content = extract_response_content_from_answer_tag(content)
    # 解析不出来response， 直接把reward 设置为 0.0
    if content is None:
        reward = 0.0
    #第一步: 基础验证
    try:
        reward = basic_verify(content, sol)
    except Exception as e:
        pass # 如果basic_verify报错, 则reward为0.0

    # 若基础验证失败, 执行第二轮验证
    if reward == 0.0:
        # 精炼
        gold_parsed = parse_latex_gt(sol)
        # 若成功精炼出 ground truth
        if len(gold_parsed) != 0:
            # 尝试从 response content 中提取
            try:
                reward = second_verify(content, sol)
            except Exception as e:
                pass # 如果basic_verify报错, 则reward为0.0

    if reward == 0.0:
        try:
            reward = extract_ans_verify(content, sol)
        except Exception as e:
            pass # 如果basic_verify报错, 则reward为0.0
    return reward

# def calculate_rewards_simple(df):
#     """简单版本：只返回 rewards 列表"""
#     #for _, row in df.iterrows():
#     output = df['output']
#     answer = df['answer']
#     rewards = list(map(lambda x: default_accuracy_reward(x, answer), output))
#     return rewards

# def calculate_win_rate(df):
#     new_rewards = df['new_rewards']
#     win_rate = sum(new_rewards)/len(new_rewards)
#     return win_rate

# def get_pos_res(df):
#     outputs = df['output']
#     new_rewards = df['new_rewards']
#     pos_res = []
#     for idx, label in enumerate(new_rewards):
#         if label == 1.0:
#             pos_case = outputs[idx]
#             pos_res.append(pos_case)
#     return pos_res

# def get_neg_res(df):
#     outputs = df['output']
#     new_rewards = df['new_rewards']
#     neg_res = []
#     for idx, label in enumerate(new_rewards):
#         if label != 1.0:
#             neg_case = outputs[idx]
#             neg_res.append(neg_case)
#     return neg_res


def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    # reward_func = extra_info["reward_func"]
    acc_score = default_accuracy_reward(solution_str, ground_truth)
    # format_score = format_reward(solution_str, reward_func)
    return acc_score 