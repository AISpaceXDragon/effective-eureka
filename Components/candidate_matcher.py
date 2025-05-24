from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from Components.para_agent import initialize_model, ConversationalAgent
import streamlit as st
import re

@dataclass
class Candidate:
    name: str
    skills: List[str]
    experience: str
    education: str
    contact: str
    raw_text: str
    match_score: float = 0.0

class CandidateMatcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chain = initialize_model()
        self.agent = ConversationalAgent(self.chain)

    def _extract_candidate_info(self, resume_text: str) -> Candidate:
        """Extract structured information from resume text"""
        # First, try to extract name and contact using regex patterns
        name = self._extract_name(resume_text)
        contact = self._extract_contact(resume_text)
        
        # Create a more specific prompt for the LLM
        prompt = f"""You are an expert resume parser. Extract the following information from this resume:

RESUME TEXT:
{resume_text}

Please extract:
1. List all technical skills and programming languages mentioned
2. List all work experience with company names and durations
3. List all educational qualifications with degrees and institutions

Format your response as:
SKILLS:
- skill1
- skill2
...

EXPERIENCE:
- Company: [name], Duration: [period], Role: [title]
...

EDUCATION:
- Degree: [name], Institution: [name], Year: [year]
...

IMPORTANT: Only list information that is explicitly mentioned in the resume. Do not make assumptions or add example information."""

        response, _ = self.agent.ask(prompt)
        
        # Parse the response
        skills = []
        experience = []
        education = []
        
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('SKILLS:'):
                current_section = 'skills'
            elif line.startswith('EXPERIENCE:'):
                current_section = 'experience'
            elif line.startswith('EDUCATION:'):
                current_section = 'education'
            elif line.startswith('- '):
                if current_section == 'skills':
                    skills.append(line[2:].strip())
                elif current_section == 'experience':
                    experience.append(line[2:].strip())
                elif current_section == 'education':
                    education.append(line[2:].strip())

        return Candidate(
            name=name,
            skills=skills,
            experience='\n'.join(experience),
            education='\n'.join(education),
            contact=contact,
            raw_text=resume_text
        )

    def _extract_name(self, text: str) -> str:
        """Extract name using regex patterns"""
        # Common patterns for names in resumes
        patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # First line with capitalized words
            r'Name:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # "Name: John Doe"
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\|',  # "John Doe |"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return "Name not found"

    def _extract_contact(self, text: str) -> str:
        """Extract contact information using regex patterns"""
        contact_info = []
        
        # Email pattern
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info.append(f"Email: {email_match.group(0)}")
        
        # Phone pattern
        phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info.append(f"Phone: {phone_match.group(0)}")
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, text)
        if linkedin_match:
            contact_info.append(f"LinkedIn: {linkedin_match.group(0)}")
        
        return '\n'.join(contact_info) if contact_info else "Contact information not found"

    def _extract_skills(self, skills_text: str) -> List[str]:
        """Extract individual skills from text"""
        skills = []
        for line in skills_text.split('\n'):
            for skill in line.split(','):
                skill = skill.strip().strip('-').strip('•').strip()
                if skill and len(skill) > 1:  # Avoid single characters
                    skills.append(skill)
        return list(set(skills))  # Remove duplicates

    def _create_jd_embedding(self, jd_text: str) -> np.ndarray:
        """Create embedding for job description"""
        return self.model.encode(jd_text)

    def _create_candidate_embedding(self, candidate: Candidate) -> np.ndarray:
        """Create embedding for candidate's information"""
        # Combine relevant information for matching
        candidate_text = f"""
        Skills: {', '.join(candidate.skills)}
        Experience: {candidate.experience}
        Education: {candidate.education}
        """
        return self.model.encode(candidate_text)

    def _calculate_similarity(self, jd_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(jd_embedding, candidate_embedding) / (
            np.linalg.norm(jd_embedding) * np.linalg.norm(candidate_embedding)
        )

    def _calculate_skill_match(self, jd_skills: List[str], candidate_skills: List[str]) -> float:
        """Calculate skill match percentage"""
        if not jd_skills or not candidate_skills:
            return 0.0
        
        # Convert to lowercase for better matching
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        # Count matching skills
        matches = sum(1 for skill in jd_skills_lower if any(
            skill in candidate_skill or candidate_skill in skill
            for candidate_skill in candidate_skills_lower
        ))
        
        return (matches / len(jd_skills)) * 100

    def match_candidates(self, jd_text: str, jd_skills: List[str], candidates: List[str]) -> List[Candidate]:
        """Match candidates against job description"""
        # Create JD embedding
        jd_embedding = self._create_jd_embedding(jd_text)
        
        # Process each candidate
        processed_candidates = []
        for resume_text in candidates:
            try:
                # Extract candidate information
                candidate = self._extract_candidate_info(resume_text)
                
                # Create candidate embedding
                candidate_embedding = self._create_candidate_embedding(candidate)
                
                # Calculate overall similarity
                similarity_score = self._calculate_similarity(jd_embedding, candidate_embedding)
                
                # Calculate skill match
                skill_match = self._calculate_skill_match(jd_skills, candidate.skills)
                
                # Combine scores (70% similarity, 30% skill match)
                candidate.match_score = (similarity_score * 0.7) + (skill_match * 0.3)
                processed_candidates.append(candidate)
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")
                continue
        
        # Sort candidates by match score
        return sorted(processed_candidates, key=lambda x: x.match_score, reverse=True)

    def format_candidate_display(self, candidate: Candidate) -> str:
        """Format candidate information for display"""
        return f"""
## {candidate.name}
**Match Score:** {candidate.match_score:.1f}%

### Contact Information
{candidate.contact}

### Skills
{chr(10).join(f"• {skill}" for skill in candidate.skills)}

### Experience
{candidate.experience}

### Education
{candidate.education}
""" 