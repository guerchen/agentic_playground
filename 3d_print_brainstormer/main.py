import os
from crewai import Agent, Task, Crew, Process, LLM


creative_guru = Agent(
    role="Creative guru",
    goal="Brainstorm and generate creative ideas to the problem at hand",
    backstory="""You're an expert inventor and creativity guru. You are able to
                give the widest range of ideas to a single problem. Your genius
                is sometimes almost frightening.""",
    allow_delegation=False,
    llm=LLM(
        model="anthropic/claude-3-sonnet-20240229",
        temperature=0.9,
    ),
    verbose=True,
)

rational_thinker = Agent(
    role="Rational thinker",
    goal="Judge whether ideas make sense for a given problem and select the best fitting idea.",
    backstory="""You're always the rational guy in the room. Always knows when an
                idea will payout or not by making some calculations in your
                head using a method you yourself devised and only you understand.""",
    allow_delegation=False,
    llm=LLM(
        model="anthropic/claude-3-sonnet-20240229",
        temperature=0.1,
    ),
    verbose=True,
)



task1 = Task(
    description="""Generate a list of 50 unique 3d printed gift ideas I can print for my
                    grandma. She is 98 years old and rides a wheelchair. She
                    doesn't enjoy reading nor use glasses.""",
    expected_output="50 bullet points, each with a 3d print idea and accompanying notes.",
    agent=creative_guru,
)

task2 = Task(
    description="""Evaluate a list of 3d printed gift ideas that I can print for my
                    grandma. She is 98 years old and rides a wheelchair. She
                    doesn't enjoy reading nor use glasses.""",
    expected_output="10 bullet points, each with a 3d print idea and accompanying notes.",
    agent=rational_thinker,
)



crew = Crew(
    agents=[creative_guru, rational_thinker],
    tasks=[task1, task2],
    process=Process.sequential,
)

crew_output = crew.kickoff()

# Accessing the crew output
print(f"Raw Output: {crew_output.raw}")
if crew_output.json_dict:
    print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
if crew_output.pydantic:
    print(f"Pydantic Output: {crew_output.pydantic}")
print(f"Tasks Output: {crew_output.tasks_output}")
print(f"Token Usage: {crew_output.token_usage}")