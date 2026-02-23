from algorithms.moe_base import PseudoExpert
import numpy as np
import random
from config.variables import HUGGING_FACE_DATA_NAME
from llm.llm_direct_query_functions import run_live_query

class LLMExpert(PseudoExpert):
    """
    Expert class that simulates LLM performance based on actual problem/answer data.
    Inherits all basic expert functionality from PseudoExpert.
    """
    
    def __init__(self, llm_name, problem_data, p=0.0, id=None):
        """
        Initialize an LLM expert.
        
        Args:
            p: True success probability (float between 0 and 1)
            llm_name: Name of the LLM this expert represents (e.g., 'phi4')
            problem_data: List of problems with correct answers and LLM responses
            id: Optional identifier for the expert
        """
        super().__init__(p, id)
        self.llm_name = llm_name
        self.problem_data = problem_data
        self.current_problem_idx = 0
        
        # Validate that the LLM name exists in the problem data
        if problem_data and llm_name not in problem_data[0]:
            raise ValueError(f"LLM {llm_name} not found in problem data")
        
    def play_expert_llm(self, huggingface_data_name=HUGGING_FACE_DATA_NAME):
        """
        Simulate an expert's action by checking if the LLM's answer matches the correct answer.
        Cycles through the problem_data sequentially.
        
        Returns:
            1 if the LLM's answer matches the correct answer, 0 otherwise
        """
        if not self.problem_data and huggingface_data_name is None:
            raise ValueError("Problem data must be set to use play_expert_llm() and live llm boolean is not selected.")
        
        # Get the current problem (with wrap-around)
        # problem = self.problem_data[self.current_problem_idx % len(self.problem_data)]
        # self.current_problem_idx += 1
        if huggingface_data_name is not None:
            is_correct = run_live_query(llm_model_name=self.llm_name)
        else:
            problem = random.choice(self.problem_data)
            self.current_problem_idx = problem['ID']

            # Get the LLM's answer
            llm_answer = problem.get(self.llm_name, None)
            if llm_answer is None:
                raise ValueError(f"LLM {self.llm_name} not found in problem data")
                
            # Check if the answer matches the correct one
            is_correct = int(str(llm_answer).lower() == str(problem['Correct']).lower())
        
        # # Update internal statistics (optional - could be done separately)
        # self.update(is_correct)
        
        return is_correct
    
    def play(self):
        return self.play_expert_llm()
    
    def get_problem_count(self):
        """Return the number of available problems in the dataset."""
        return len(self.problem_data)
    
    def __str__(self):
        return (f"LLMExpert(id={self.id}, llm={self.llm_name}, p={self.p:.3f}, "
                f"est={self.get_estimate():.3f}, trials={self.trials}, "
                f"problems={self.get_problem_count()})")