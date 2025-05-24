from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from Components.para_agent import initialize_model, ConversationalAgent
import streamlit as st
import re
from datetime import datetime
import dateutil.parser
from pyresparser import ResumeParser
import os
import tempfile
import json
from pathlib import Path

@dataclass
class Candidate:
    name: str
    skills: List[str]
    experience: str
    total_experience_years: float
    highest_education: str
    contact: str
    raw_text: str
    match_score: float = 0.0
    skill_match_percentage: float = 0.0
    experience_match_score: float = 0.0
    education_match_score: float = 0.0

class CandidateMatcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chain = initialize_model()
        self.agent = ConversationalAgent(self.chain)
        
        # Create custom config for pyresparser
        self._create_pyresparser_config()
        
        # Education level mapping
        self.education_levels = {
            'phd': 5,
            'doctorate': 5,
            'master': 4,
            'm.tech': 4,
            'm.eng': 4,
            'bachelor': 3,
            'b.tech': 3,
            'b.eng': 3,
            'diploma': 2,
            'high school': 1
        }

        # Skill categories for better matching
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite'],
            'ml_ai': ['machine learning', 'deep learning', 'ai', 'artificial intelligence', 'tensorflow', 'pytorch', 'keras', 'scikit-learn'],
            'data_science': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'r', 'spark', 'hadoop'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'aws', 'azure', 'gcp', 'terraform'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'xamarin'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'serverless', 'lambda', 's3', 'ec2'],
            'security': ['security', 'cybersecurity', 'penetration testing', 'ethical hacking', 'network security'],
            'blockchain': ['blockchain', 'ethereum', 'solidity', 'web3', 'smart contracts']
        }

    def _create_pyresparser_config(self):
        """Create custom config file for pyresparser"""
        config = {
            "skills": {
                "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift", "kotlin"],
                "web": ["html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "spring"],
                "database": ["sql", "mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite"],
                "ml_ai": ["machine learning", "deep learning", "ai", "artificial intelligence", "tensorflow", "pytorch", "keras", "scikit-learn"],
                "data_science": ["pandas", "numpy", "matplotlib", "seaborn", "jupyter", "r", "spark", "hadoop"],
                "devops": ["docker", "kubernetes", "jenkins", "git", "ci/cd", "aws", "azure", "gcp", "terraform"],
                "mobile": ["android", "ios", "react native", "flutter", "xamarin"],
                "cloud": ["aws", "azure", "gcp", "cloud", "serverless", "lambda", "s3", "ec2"],
                "security": ["security", "cybersecurity", "penetration testing", "ethical hacking", "network security"],
                "blockchain": ["blockchain", "ethereum", "solidity", "web3", "smart contracts"]
            }
        }
        
        # Create config directory if it doesn't exist
        config_dir = Path.home() / '.pyresparser'
        config_dir.mkdir(exist_ok=True)
        
        # Write config file
        config_file = config_dir / 'config.cfg'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def _calculate_experience_years(self, experience_text: str) -> float:
        """Calculate total years of experience from experience text"""
        total_years = 0.0
        experience_entries = experience_text.split('\n')
        
        for entry in experience_entries:
            # Extract duration using regex
            duration_match = re.search(r'Duration:\s*([^-]+)\s*-\s*([^,]+)', entry)
            if duration_match:
                try:
                    start_date = dateutil.parser.parse(duration_match.group(1).strip())
                    end_date = dateutil.parser.parse(duration_match.group(2).strip())
                    
                    # Calculate years between dates
                    years = (end_date - start_date).days / 365.25
                    if years > 0:  # Only add positive durations
                        total_years += years
                except:
                    continue
        
        return round(total_years, 1)

    def _get_highest_education(self, education_text: str) -> str:
        """Determine the highest level of education"""
        highest_level = 0
        highest_degree = ""
        
        for line in education_text.split('\n'):
            degree = line.lower()
            for level, score in self.education_levels.items():
                if level in degree and score > highest_level:
                    highest_level = score
                    highest_degree = line.strip()
        
        return highest_degree if highest_degree else "Education level not found"

    def _calculate_skill_match_score(self, jd_skills: List[str], candidate_skills: List[str]) -> Tuple[float, List[str]]:
        """Calculate detailed skill match score and matching skills"""
        if not jd_skills or not candidate_skills:
            return 0.0, []
        
        # Convert to lowercase for better matching
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        # Find matching skills using category-based matching
        matching_skills = []
        for jd_skill in jd_skills_lower:
            # Check direct match
            if jd_skill in candidate_skills_lower:
                matching_skills.append(jd_skill)
                continue
            
            # Check category-based match
            for category, skills in self.skill_categories.items():
                if jd_skill in skills:
                    # If JD skill is in a category, check if candidate has any skill from that category
                    for candidate_skill in candidate_skills_lower:
                        if candidate_skill in skills:
                            matching_skills.append(candidate_skill)
                            break
                    break
        
        # Calculate match percentage
        match_percentage = (len(matching_skills) / len(jd_skills)) * 100
        
        return match_percentage, matching_skills

    def _calculate_experience_match_score(self, required_years: int, candidate_years: float) -> float:
        """Calculate experience match score"""
        if required_years == 0:
            return 100.0
        
        # Calculate match percentage based on experience
        if candidate_years >= required_years:
            return 100.0
        else:
            return (candidate_years / required_years) * 100

    def _extract_candidate_info(self, resume_text: str, file_path: str = None) -> Candidate:
        """Extract structured information from resume using pyresparser"""
        try:
            # Save resume text to a temporary file if file_path is not provided
            if not file_path:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(resume_text)
                    file_path = temp_file.name

            # Parse resume using pyresparser
            data = ResumeParser(file_path).get_extracted_data()
            
            # Extract information
            name = data.get('name', 'Name not found')
            skills = data.get('skills', [])
            experience = data.get('experience', [])
            education = data.get('education', [])
            
            # Format experience
            formatted_experience = []
            for exp in experience:
                if isinstance(exp, dict):
                    company = exp.get('company', '')
                    duration = exp.get('duration', '')
                    role = exp.get('title', '')
                    formatted_experience.append(f"Company: {company}, Duration: {duration}, Role: {role}")
                else:
                    formatted_experience.append(str(exp))
            
            # Calculate total experience
            total_experience = self._calculate_experience_years('\n'.join(formatted_experience))
            
            # Get highest education
            highest_education = self._get_highest_education('\n'.join(education))
            
            # Format contact information
            contact_info = []
            if data.get('email'):
                contact_info.append(f"Email: {data['email']}")
            if data.get('mobile_number'):
                contact_info.append(f"Phone: {data['mobile_number']}")
            if data.get('linkedin'):
                contact_info.append(f"LinkedIn: {data['linkedin']}")
            
            contact = '\n'.join(contact_info) if contact_info else "Contact information not found"
            
            # Clean up temporary file if created
            if not file_path.endswith('.pdf'):
                os.unlink(file_path)
            
            return Candidate(
                name=name,
                skills=skills,
                experience='\n'.join(formatted_experience),
                total_experience_years=total_experience,
                highest_education=highest_education,
                contact=contact,
                raw_text=resume_text
            )
            
        except Exception as e:
            st.error(f"Error parsing resume with pyresparser: {str(e)}")
            # Fallback to previous method if pyresparser fails
            return self._extract_candidate_info_fallback(resume_text)

    def _extract_candidate_info_fallback(self, resume_text: str) -> Candidate:
        """Fallback method using LLM if pyresparser fails"""
        # First, try to extract name and contact using regex patterns
        name = self._extract_name(resume_text)
        contact = self._extract_contact(resume_text)
        
        # Create a more specific prompt for the LLM
        prompt = f"""You are an expert resume parser. Extract the following information from this resume:

RESUME TEXT:
{resume_text}

Please extract:
1. List all technical skills and programming languages mentioned
2. List all work experience with company names and durations (include exact dates)
3. List all educational qualifications with degrees and institutions

Format your response as:
SKILLS:
- skill1
- skill2
...

EXPERIENCE:
- Company: [name], Duration: [start date] - [end date], Role: [title]
...

EDUCATION:
- Degree: [name], Institution: [name], Year: [year]
...

IMPORTANT: 
- Only list information that is explicitly mentioned in the resume
- For experience, include exact dates in format: Month Year - Month Year
- For education, include the year of completion
- Do not make assumptions or add example information"""

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

        # Calculate total experience and highest education
        total_experience = self._calculate_experience_years('\n'.join(experience))
        highest_education = self._get_highest_education('\n'.join(education))

        return Candidate(
            name=name,
            skills=skills,
            experience='\n'.join(experience),
            total_experience_years=total_experience,
            highest_education=highest_education,
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
        Education: {candidate.highest_education}
        """
        return self.model.encode(candidate_text)

    def _calculate_similarity(self, jd_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(jd_embedding, candidate_embedding) / (
            np.linalg.norm(jd_embedding) * np.linalg.norm(candidate_embedding)
        )

    def match_candidates(self, jd_text: str, jd_skills: List[str], required_experience: int, candidates: List[str]) -> List[Candidate]:
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
                skill_match, matching_skills = self._calculate_skill_match_score(jd_skills, candidate.skills)
                candidate.skill_match_percentage = skill_match
                
                # Calculate experience match
                experience_match = self._calculate_experience_match_score(required_experience, candidate.total_experience_years)
                candidate.experience_match_score = experience_match
                
                # Combine scores (40% similarity, 40% skill match, 20% experience match)
                candidate.match_score = (
                    similarity_score * 0.4 +
                    skill_match * 0.4 +
                    experience_match * 0.2
                )
                
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

### Skills Match
**Skill Match Percentage:** {candidate.skill_match_percentage:.1f}%
**Total Experience:** {candidate.total_experience_years} years
**Highest Education:** {candidate.highest_education}

### Skills
{chr(10).join(f"• {skill}" for skill in candidate.skills)}

### Experience
{candidate.experience}

### Education
{candidate.highest_education}
""" 