from dataclasses import dataclass
from typing import List, Optional
from Components.para_agent import initialize_model, ConversationalAgent

@dataclass
class JDInput:
    """Structured input for job description generation"""
    role: str
    experience_years: int
    required_skills: List[str]
    location: Optional[str] = None
    company_type: Optional[str] = None
    industry: Optional[str] = None
    salary_range: Optional[str] = None
    additional_requirements: Optional[str] = None

@dataclass
class JDOutput:
    """Structured output for job description"""
    job_title: str
    overview: str
    key_responsibilities: List[str]
    required_skills: List[str]
    preferred_qualifications: List[str]
    benefits: List[str]
    additional_info: Optional[str] = None

class JDGenerator:
    def __init__(self):
        self.chain = initialize_model([])  # Initialize without documents
        self.agent = ConversationalAgent(self.chain)

    def _create_prompt(self, jd_input: JDInput) -> str:
        """Create a structured prompt for the LLM"""
        prompt = f"""Generate a detailed job description for a {jd_input.role} position.

Job Details:
- Experience Required: {jd_input.experience_years} years
- Required Skills: {', '.join(jd_input.required_skills)}
"""

        if jd_input.location:
            prompt += f"- Location: {jd_input.location}\n"
        if jd_input.company_type:
            prompt += f"- Company Type: {jd_input.company_type}\n"
        if jd_input.industry:
            prompt += f"- Industry: {jd_input.industry}\n"
        if jd_input.salary_range:
            prompt += f"- Salary Range: {jd_input.salary_range}\n"
        if jd_input.additional_requirements:
            prompt += f"- Additional Requirements: {jd_input.additional_requirements}\n"

        prompt += """
Please generate a comprehensive job description with the following sections:
1. Job Title and Overview
2. Key Responsibilities (list 5-7 points)
3. Required Skills and Qualifications
4. Preferred Qualifications
5. Company Benefits

Format the output as a JSON object with the following structure:
{
    "job_title": "string",
    "overview": "string",
    "key_responsibilities": ["string"],
    "required_skills": ["string"],
    "preferred_qualifications": ["string"],
    "benefits": ["string"],
    "additional_info": "string"
}
"""
        return prompt

    def _parse_llm_response(self, response: str) -> JDOutput:
        """Parse the LLM response into structured format"""
        try:
            # The response should be a JSON string
            import json
            data = json.loads(response)
            
            return JDOutput(
                job_title=data.get("job_title", ""),
                overview=data.get("overview", ""),
                key_responsibilities=data.get("key_responsibilities", []),
                required_skills=data.get("required_skills", []),
                preferred_qualifications=data.get("preferred_qualifications", []),
                benefits=data.get("benefits", []),
                additional_info=data.get("additional_info", "")
            )
        except Exception as e:
            # If parsing fails, return a formatted string version
            return JDOutput(
                job_title="",
                overview=response,
                key_responsibilities=[],
                required_skills=[],
                preferred_qualifications=[],
                benefits=[],
                additional_info=""
            )

    def generate_jd(self, jd_input: JDInput) -> JDOutput:
        """Generate a job description based on input parameters"""
        prompt = self._create_prompt(jd_input)
        response, _ = self.agent.ask(prompt)
        return self._parse_llm_response(response)

    def format_jd_for_display(self, jd_output: JDOutput) -> str:
        """Format the JD output for display"""
        formatted_text = f"""
# {jd_output.job_title}

## Overview
{jd_output.overview}

## Key Responsibilities
{chr(10).join(f"- {resp}" for resp in jd_output.key_responsibilities)}

## Required Skills
{chr(10).join(f"- {skill}" for skill in jd_output.required_skills)}

## Preferred Qualifications
{chr(10).join(f"- {qual}" for qual in jd_output.preferred_qualifications)}

## Benefits
{chr(10).join(f"- {benefit}" for benefit in jd_output.benefits)}
"""
        if jd_output.additional_info:
            formatted_text += f"\n## Additional Information\n{jd_output.additional_info}"

        return formatted_text 