from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# Initialize LLM
llm = ChatOllama(model="phi3:mini")


# Prompt Template (Prompt Engineering)
prompt = PromptTemplate(
    template="""
You are an experienced technical recruiter evaluating resumes.

Your task is to analyze a candidate resume for a specific job role and provide structured feedback.

Follow this format exactly:

Candidate Summary:
<short summary of the candidate>

Matching Skills:
<list of skills from the resume that match the job role>

Missing Skills:
<skills required for the job role but missing in the resume>

Experience Evaluation:
<evaluate whether the experience matches the role>

Recommendation:
<Strong Fit / Moderate Fit / Weak Fit>

Resume:
{resume}

Job Role:
{job_role}
""",
    input_variables=["resume", "job_role"],
)


print("AI Resume Reviewer")
print("Type 'exit' to quit")


while True:

    resume_text = input("\nEnter Resume Text:\n")

    if resume_text.lower() == "exit":
        break

    job_role = input("\nEnter Job Role:\n")

    formatted_prompt = prompt.format(
        resume=resume_text,
        job_role=job_role
    )

    response = llm.invoke(formatted_prompt)

    print("\nResume Evaluation:\n")
    print(response.content)
