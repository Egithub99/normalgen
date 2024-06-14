import autogen
from autogen import Skill

# Define the configuration for the agents
gemma = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

# Initialize the TimberExpert agent
timber_expert_agent = autogen.AssistantAgent(
    name="TimberExpert",
    llm_config=gemma,
    system_message="""
    You are a highly skilled structural engineer with specialized expertise in timber. Your role is to evaluate whether timber is an appropriate material for given structural engineering problems. You consider factors such as load-bearing capacity, environmental conditions, durability, cost, and sustainability when making your assessment. Provide detailed explanations for your recommendations and cite relevant engineering standards or research where applicable.
    """
)

# Define the skill
@Skill
def evaluate_timber_suitability(problem_description):
    """
    Evaluate whether timber is an appropriate material for a given structural engineering problem.
    
    Parameters:
    problem_description (str): A detailed description of the structural engineering problem, including
    requirements such as load-bearing capacity, environmental conditions, expected lifespan, cost considerations,
    and any other relevant factors.
    
    Returns:
    dict: A dictionary containing the evaluation results, with keys such as 'suitability', 'explanation', 
    and 'references'. The 'suitability' key will have a boolean value indicating if timber is suitable. The 
    'explanation' key will contain a detailed explanation of the assessment, and the 'references' key will 
    list any relevant standards or research papers cited in the evaluation.
    
    Example:
        problem_description = "A residential building in a seismic zone requiring high load-bearing capacity 
        and durability."
        results = evaluate_timber_suitability(problem_description)
        print(results)
    """
    # Here you would include the actual evaluation logic, which might involve complex calculations,
    # database lookups for material properties, or consultations with standards and literature.
    
    # For demonstration purposes, we'll return a simple mock result.
    results = {
        "suitability": False,
        "explanation": "Timber is not suitable for a parking garage due to its lack of robustness and fire resistance.",
        "references": [
            "ISO 21581: Timber structures — Static and cyclic lateral load test method for shear walls",
            "FEMA P-807: Guidelines for the Seismic Retrofit of Soft-Story Wood-frame Buildings"
        ]
    }
    return results

# Register the skill with the agent
timber_expert_agent.add_skill(evaluate_timber_suitability)

# Define the workflow steps
def workflow():
    # Step 1: User describes the structural engineering problem
    problem_description = input("Describe the structural engineering problem: ")
    
    # Step 2: TimberExpert evaluates the problem
    evaluation_results = timber_expert_agent.call(
        skill_name="evaluate_timber_suitability",
        inputs=problem_description
    )
    
    # Step 3: TimberExpert returns the evaluation
    print("Evaluation Results:")
    print(evaluation_results)

# Execute the workflow
if __name__ == "__main__":
    workflow()
