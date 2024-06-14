import autogen
from fpdf import FPDF

# Configuration list for the language model
llm_config = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}


# Initialize the agent
problem_analysis_agent = autogen.AssistantAgent(
    name="ProblemAnalysisAgent",
    llm_config=llm_config
)

def create_project_brief_pdf(client_requirements, design_brief):
    """
    This function creates a PDF document containing the project brief.

    Parameters:
    - client_requirements: str: Detailed client requirements for the project.
    - design_brief: str: Initial design brief for the project.

    Returns:
    - None
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Project Brief", ln=True, align='C')

    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt=f"Client Requirements:\n{client_requirements}\n\nDesign Brief:\n{design_brief}")

    pdf.output("Project_Brief.pdf")

def analyze_problem(user_query):
    """
    This function analyzes the problem based on the user's query and performs the following tasks:
    - Contributes to the preparation of Client Requirements (Stage 0)
    - Contributes to the preparation of the design brief (Stage 1)
    
    Parameters:
    - user_query: str: The user's query describing the project.
    
    Returns:
    - None
    """
    # Simulate the analysis process (placeholder)
    client_requirements = "Client wants a multi-level parking garage with a capacity of 200 cars, automated ticketing system, and security surveillance."
    design_brief = "The project involves designing a multi-level parking garage with specified features and security measures."

    # Create a PDF document containing the project brief
    create_project_brief_pdf(client_requirements, design_brief)

# Define the workflow
def workflow():
    user_query = input("Enter the project query: ")
    problem_analysis_agent.execute(
        task="Analyze the problem based on the project query and prepare the initial project brief",
        context=user_query
    )
    analyze_problem(user_query)

# Execute the workflow
workflow()
