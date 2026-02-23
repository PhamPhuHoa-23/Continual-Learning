from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import re
import re
from typing import Optional

# Define question formatting functions

def format_for_llm_gsm8k(example):
    """Formats a GSM8K example with strict instructions for numerical output"""
    question = example['question']
    answer = example['answer'].split('#### ')[-1].strip()
    
    # Create a strict prompt that forces numerical output only
    llm_input = (
        f"Question: {question}"
    )
    
    return {
        "question": llm_input,
        "correct_answer": answer
    }

def format_for_llm_commonsense(example):
    """Formats a commonsense example with strict instructions for numerical output"""
    question = example['question']
    choices = example['choices']
    answer = example['answerKey']
    
    
    return {
        "question": question,
        "choices": choices,
        "correct_answer": answer
    }

def format_for_llm_boolq(example):
    """Formats a boolq example with strict instructions for numerical output"""
    question = example['question']
    answer = example['answer']
        
    return {
        "question": question,
        "correct_answer": answer
    }

# Define prompting logic

def get_short_answer_from_llm_gsm8k(query_dic, llm_model="dolphin-phi") -> str:

    question_str = query_dic["question"]
    template = """
    ### Instructions:
    1. Read the question carefully and identify what is being asked.
    2. Solve the problem methodically, showing each step clearly.
    3. Double-check your calculations llm_query_func = get_short_answer_from_llmbefore finalizing the answer.
    4. Your final output MUST follow EXACTLY this format:
    
    ### Reasoning:
    [Your step-by-step reasoning here]
    
    ### Final Answer: [Numerical Value]

    ### Required Output Format Rules:
    - Only numbers are allowed in the final answer (e.g., 42, 3.14, 2/3)
    - If you cannot determine the answer, you MUST write: ### Final Answer: 0
    - No additional text, explanations, or characters after the final answer
    - The final answer line must be the very last line of your response

    ### Question:
    {question}

    ### Reasoning:
    """
    # - If unsure, return `### Final Answer: 0`.

    evaluation_prompt = ChatPromptTemplate.from_template(template)
    evaluation_llm = evaluation_prompt | OllamaLLM(model=llm_model)
    evaluation_result = evaluation_llm.invoke({"question": question_str})
    
    # Extract the final answer using stricter parsing
    final_answer_match = re.search(
        r'### Final Answer:\s*(-?\d+\.?\d*|[-+]?\d+/\d+)',
        evaluation_result, 
        re.IGNORECASE
    )
    
    if final_answer_match:
        return final_answer_match.group(1).strip()
    return extract_answer_with_llm(question_str, llm_model=llm_model)  # Fallback to v2 if not found
    # return "0"  # Fallback


def extract_answer_with_llm(question_str: str, llm_model="dolphin-phi") -> str:
    
    template = """
    {question}
    """
    evaluation_prompt = ChatPromptTemplate.from_template(template)
    evaluation_llm = evaluation_prompt | OllamaLLM(model=llm_model)
    evaluation_result = evaluation_llm.invoke({"question": question_str})
    
    # Extract the final answer using stricter parsing
    final_answer_match = re.search(
        r'### Final Answer:\s*(-?\d+\.?\d*|[-+]?\d+/\d+)', 
        evaluation_result, 
        re.IGNORECASE
    )

    ### Phi4 Extraction
    answer_extract_template = """
    Find the final answer in this text, and return ONLY the final answer as ### Final Answer: X
    ### Final Answer: [Numerical Value]

    ### Answer:
    {evaluation_result}
    """

    answer_extract_evaluation_prompt = ChatPromptTemplate.from_template(answer_extract_template)
    answer_extract_evaluation_llm = answer_extract_evaluation_prompt | OllamaLLM(model="phi4")
    evaluation_result = answer_extract_evaluation_llm.invoke({"evaluation_result": evaluation_result})

    if final_answer_match:
        return final_answer_match.group(1).strip()
    return "0"  # Fallback


def get_short_answer_from_llm_commonsense(query_dic, llm_model="dolphin-phi") -> str:
    template = """
    ### Instructions:
    1. Read the question carefully and identify what is being asked.
    2. Solve the problem methodically, showing each step clearly.
    3. Double-check your calculations before finalizing the answer.
    4. Your final output MUST follow EXACTLY this format:
    
    ### Reasoning:
    [Your step-by-step reasoning here]
    
    ### Final Answer: One of 5 catagories [A, B, C, D, E]

    ### Required Output Format Rules:
    - Only numbers are allowed in the final answer (e.g., A, B, C, D, E)
    - If you cannot determine the answer, you MUST pick a random answer from [A, B, C, D, E].
    - No additional text, explanations, or characters after the final answer.
    - The final answer line must be the very last line of your response.

    ### Question:
    {question}

    ### Choices:
    {choices}

    ### Reasoning:
    """

    question_str = query_dic['question']
    choices_str = str(query_dic['choices'])

    evaluation_prompt = ChatPromptTemplate.from_template(template)
    evaluation_llm = evaluation_prompt | OllamaLLM(model=llm_model)
    evaluation_result = evaluation_llm.invoke({"question": question_str, "choices": choices_str})
    
    # Extract the final answer using stricter parsing
    final_answer_match = re.search(
        r'### Final Answer:\s*([A-Ea-e])', 
        evaluation_result, 
        re.IGNORECASE
    )
    
    if final_answer_match:
        return final_answer_match.group(1).strip()
    return "A"

def get_short_answer_from_llm_boolq(query_dic, llm_model="dolphin-phi") -> str:
    template = """
    ### Instructions:
    1. Read the question carefully and identify what is being asked.
    2. Solve the problem methodically, showing each step clearly.
    3. Double-check your calculations before finalizing the answer.
    4. Your final output MUST follow EXACTLY this format:
    
    ### Reasoning:
    [Your step-by-step reasoning here]
    
    ### Final Answer: One of 2 catagories [true, false]

    ### Required Output Format Rules:
    - Only numbers are allowed in the final answer (e.g., true, false)
    - If you cannot determine the answer, you MUST pick a random answer from [true, false].
    - No additional text, explanations, or characters after the final answer.
    - The final answer line must be the very last line of your response.

    ### Question:
    {question}

    ### Reasoning:
    """

    question_str = query_dic['question']

    evaluation_prompt = ChatPromptTemplate.from_template(template)
    evaluation_llm = evaluation_prompt | OllamaLLM(model=llm_model)
    evaluation_result = evaluation_llm.invoke({"question": question_str})
    
    # Extract the final answer using stricter parsing
    final_answer_match = re.search(
        r'### Final Answer:\s*(true|false)[\W]?', 
        evaluation_result, 
        re.IGNORECASE
    )
    
    if final_answer_match:
        return final_answer_match.group(1).strip()
    return "A"