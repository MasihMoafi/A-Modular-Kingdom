from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from database import Database
from llm_config import get_llm
import json
from typing import Dict, Any

class GymCrew:
    def __init__(self):
        self.db = Database()
        self.llm = get_llm()
        self.agents = self._create_agents()
        self.user_states = {}  # Track conversation state per user
    
    def _create_agents(self):
        interviewer = Agent(
            role='Fitness Interviewer',
            goal='Conduct comprehensive fitness assessments to understand user needs',
            backstory='Expert fitness consultant who asks the right questions to create personalized fitness plans',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        plan_generator = Agent(
            role='Workout Plan Generator',
            goal='Create detailed, progressive workout plans based on user profiles',
            backstory='Certified personal trainer with expertise in exercise science and progressive overload',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        nutrition_agent = Agent(
            role='Nutrition Advisor',
            goal='Provide personalized nutrition guidance to support fitness goals',
            backstory='Registered dietitian specializing in sports nutrition and meal planning',
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        return {
            'interviewer': interviewer,
            'plan_generator': plan_generator,
            'nutrition_agent': nutrition_agent
        }
    
    def start_user_journey(self, user_id: str) -> str:
        """Start the interview process for a new user"""
        self.user_states[user_id] = {
            'stage': 'interviewing',
            'interview_data': {},
            'questions_asked': 0
        }
        
        interview_task = Task(
            description=f"""
            Start a fitness interview for user {user_id}. Ask about:
            1. Current fitness level (beginner/intermediate/advanced)
            2. Fitness goals (weight loss, muscle gain, endurance, strength)
            3. Available equipment (home gym, commercial gym, bodyweight only)
            4. Time availability (days per week, minutes per session)
            5. Injuries or limitations
            
            Ask ONE question at a time in a friendly, conversational manner.
            Start with fitness level assessment.
            """,
            agent=self.agents['interviewer'],
            expected_output="A single, engaging question about the user's fitness level"
        )
        
        crew = Crew(
            agents=[self.agents['interviewer']],
            tasks=[interview_task],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        # Save conversation
        self.db.save_conversation(user_id, 'interviewer', 'start_interview', str(result))
        
        return str(result)
    
    def process_user_response(self, user_id: str, response: str) -> Dict[str, Any]:
        """Process user response and determine next action"""
        if user_id not in self.user_states:
            return {"error": "User session not found"}
        
        user_state = self.user_states[user_id]
        
        if user_state['stage'] == 'interviewing':
            return self._continue_interview(user_id, response)
        elif user_state['stage'] == 'plan_generation':
            return self._generate_workout_plan(user_id)
        else:
            return {"error": "Unknown stage"}
    
    def _continue_interview(self, user_id: str, response: str) -> Dict[str, Any]:
        """Continue the interview process"""
        user_state = self.user_states[user_id]
        questions_asked = user_state['questions_asked']
        
        # Store the response
        question_types = ['fitness_level', 'goals', 'equipment', 'time_availability', 'injuries']
        if questions_asked < len(question_types):
            user_state['interview_data'][question_types[questions_asked]] = response
        
        user_state['questions_asked'] += 1
        
        # Check if interview is complete
        if user_state['questions_asked'] >= 5:
            # Save complete profile
            self.db.save_user_profile(user_id, user_state['interview_data'])
            user_state['stage'] = 'plan_generation'
            return self._generate_workout_plan(user_id)
        
        # Ask next question
        next_question_task = Task(
            description=f"""
            Continue the fitness interview for user {user_id}.
            Previous response: {response}
            Questions asked so far: {user_state['questions_asked']}
            
            Ask the next appropriate question based on this sequence:
            1. Fitness level ✓
            2. Goals
            3. Equipment
            4. Time availability
            5. Injuries/limitations
            
            Ask ONE question at a time in a friendly manner.
            """,
            agent=self.agents['interviewer'],
            expected_output="The next interview question"
        )
        
        crew = Crew(
            agents=[self.agents['interviewer']],
            tasks=[next_question_task],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        # Save conversation
        self.db.save_conversation(user_id, 'interviewer', response, str(result))
        
        return {"message": str(result), "stage": "interviewing"}
    
    def _generate_workout_plan(self, user_id: str) -> Dict[str, Any]:
        """Generate workout plan based on user profile"""
        profile = self.db.get_user_profile(user_id)
        if not profile:
            return {"error": "User profile not found", "stage": "error"}
        
        plan_task = Task(
            description=f"""
            You are creating a personalized workout plan for a user. Based on their profile:
            - Fitness level: {profile.get('fitness_level', 'Unknown')}
            - Goals: {profile.get('goals', 'Unknown')}
            - Equipment: {profile.get('equipment', 'Unknown')}
            - Time availability: {profile.get('time_availability', 'Unknown')}
            - Injuries/limitations: {profile.get('injuries', 'None specified')}
            
            Create a conversational, encouraging response that includes:
            1. A personalized greeting acknowledging their goals
            2. A clear weekly workout schedule
            3. Specific exercises with sets/reps
            4. Progressive overload explanation
            5. Modifications for their equipment/limitations
            
            Be encouraging and professional. Format as readable text, not JSON.
            """,
            agent=self.agents['plan_generator'],
            expected_output="A friendly, detailed workout plan explanation"
        )
        
        nutrition_task = Task(
            description=f"""
            You are providing nutrition guidance to complement the workout plan. Based on their profile:
            - Goals: {profile.get('goals', 'Unknown')}
            - Fitness level: {profile.get('fitness_level', 'Unknown')}
            
            Create a conversational response that includes:
            1. Daily calorie guidance
            2. Macronutrient recommendations
            3. Meal timing tips
            4. Hydration advice
            5. Practical food suggestions
            
            Be supportive and make it actionable. Format as readable text, not JSON.
            """,
            agent=self.agents['nutrition_agent'],
            expected_output="Friendly nutrition guidance and recommendations"
        )
        
        crew = Crew(
            agents=[self.agents['plan_generator'], self.agents['nutrition_agent']],
            tasks=[plan_task, nutrition_task],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        # Save workout plan
        plan_data = {"workout": str(result), "profile": profile}
        workout_id = self.db.save_workout_plan(user_id, plan_data)
        
        # Update user state
        self.user_states[user_id]['stage'] = 'done'
        self.user_states[user_id]['workout_id'] = workout_id
        
        return {"message": str(result), "stage": "plan_generated", "workout_id": workout_id}
