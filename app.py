import streamlit as st
import json
import os
from utils.llm_api import generate_answer, grade_answer
from utils.plag_checker import check_plagiarism
import uuid
from datetime import datetime

# Initialize session state
if 'assessments' not in st.session_state:
    st.session_state.assessments = {}
if 'current_assessment' not in st.session_state:
    st.session_state.current_assessment = None
if 'student_answers' not in st.session_state:
    st.session_state.student_answers = {}

def load_assessments():
    """Load assessments from JSON file"""
    try:
        if os.path.exists('assessments.json'):
            with open('assessments.json', 'r') as f:
                data = json.load(f)
                st.session_state.assessments = data.get('assessments', {})
    except Exception as e:
        st.error(f"Error loading assessments: {str(e)}")

def save_assessments():
    """Save assessments to JSON file"""
    try:
        data = {
            'assessments': st.session_state.assessments,
            'last_updated': datetime.now().isoformat()
        }
        with open('assessments.json', 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving assessments: {str(e)}")
        return False

def create_assessment_page():
    """Teacher view for creating assessments"""
    st.header("Create Assessment")
    
    # Assessment title (use session_state if set by link)
    assessment_title = st.text_input("Assessment Title", placeholder="Enter assessment title", value=st.session_state.get('assessment_title', ''))
    
    if not assessment_title:
        st.info("Please enter an assessment title to continue.")
        return
    
    # Initialize assessment in session state if not exists
    if assessment_title not in st.session_state.assessments:
        st.session_state.assessments[assessment_title] = {
            'id': str(uuid.uuid4()),
            'title': assessment_title,
            'questions': [],
            'total_marks': 0,
            'created_at': datetime.now().isoformat()
        }
    
    current_assessment = st.session_state.assessments[assessment_title]
    
    st.subheader("Add Questions")
    
    # Initialize form data in session state
    if 'form_question' not in st.session_state:
        st.session_state.form_question = ""
    if 'form_marks' not in st.session_state:
        st.session_state.form_marks = 5
    if 'form_model_answer' not in st.session_state:
        st.session_state.form_model_answer = ""
    if 'form_word_limit' not in st.session_state:
        st.session_state.form_word_limit = 0
    
    # Question input form
    question_text = st.text_area("Question Text", 
                                value=st.session_state.form_question,
                                placeholder="Enter your question here...")
    marks = st.number_input("Marks", min_value=1, max_value=100, 
                           value=st.session_state.form_marks)
    model_answer = st.text_area("Answer", 
                               value=st.session_state.form_model_answer,
                               placeholder="Write the answer or generate it with AI...")
    word_limit = st.number_input("Minimum Word Limit (0 = No Limit)", min_value=0, max_value=1000, value=st.session_state.form_word_limit, help="Set 0 for no minimum word limit.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        generate_answer_btn = st.button("Generate Answer with AI", help="Use AI to create a model answer")
    with col2:
        submit_question = st.button("Submit Question", type="primary", help="Add this question to the assessment")
    with col3:
        clear_form = st.button("Clear Form", help="Reset the form fields")
    
    # Handle generate answer
    if generate_answer_btn and question_text:
        with st.spinner("Generating answer with AI..."):
            generated_answer = generate_answer(question_text)
            if generated_answer:
                st.session_state.form_model_answer = generated_answer
                st.success("Answer generated successfully!")
                st.session_state['page'] = "Create Assessment"
                st.session_state['assessment_title'] = assessment_title
                st.rerun()
            else:
                st.error("Failed to generate answer. Please try again or enter manually.")
    
    # Handle clear form
    if clear_form:
        st.session_state.form_question = ""
        st.session_state.form_marks = 5
        st.session_state.form_model_answer = ""
        st.session_state.form_word_limit = 0
        st.session_state['page'] = "Create Assessment"
        st.session_state['assessment_title'] = assessment_title
        st.rerun()
    
    # Update session state with current values
    st.session_state.form_question = question_text
    st.session_state.form_marks = marks
    st.session_state.form_model_answer = model_answer
    st.session_state.form_word_limit = word_limit
    
    # Handle submit question
    if submit_question:
        if not question_text.strip():
            st.error("Please enter a question before submitting.")
        else:
            question_data = {
                'id': str(uuid.uuid4()),
                'text': question_text.strip(),
                'marks': marks,
                'model_answer': model_answer.strip() if model_answer else "",
                'word_limit': word_limit,
                'created_at': datetime.now().isoformat()
            }
            
            current_assessment['questions'].append(question_data)
            current_assessment['total_marks'] += marks
            
            if save_assessments():
                st.success(f"✅ Question submitted successfully! Running total: {current_assessment['total_marks']} marks")
                # Clear form after successful addition
                st.session_state.form_question = ""
                st.session_state.form_marks = 5
                st.session_state.form_model_answer = ""
                st.session_state.form_word_limit = 0
                st.session_state['page'] = "Create Assessment"
                st.session_state['assessment_title'] = assessment_title
                st.rerun()
            else:
                st.error("Failed to save question. Please try again.")
    
    # Display current questions
    if current_assessment['questions']:
        st.divider()
        st.subheader(f"📝 Questions Added ({len(current_assessment['questions'])})")
        st.info(f"**Current Total: {current_assessment['total_marks']} marks**")
        
        for i, question in enumerate(current_assessment['questions']):
            with st.expander(f"Question {i+1} ({question['marks']} marks)", expanded=False):
                st.markdown(f"**Question:** {question['text']}")
                if question['model_answer']:
                    st.markdown(f"**Model Answer:** {question['model_answer']}")
                else:
                    st.markdown("**Model Answer:** Not provided")
                
                # Option to delete question
                if st.button(f"🗑️ Remove Question {i+1}", key=f"delete_{question['id']}", help="Delete this question from the assessment"):
                    current_assessment['questions'].pop(i)
                    current_assessment['total_marks'] -= question['marks']
                    save_assessments()
                    st.success(f"Question {i+1} removed successfully!")
                    st.session_state['page'] = "Create Assessment"
                    st.session_state['assessment_title'] = assessment_title
                    st.rerun()
    
    # Final assessment submission
    if current_assessment['questions']:
        st.divider()
        st.subheader("📋 Complete Assessment")
        st.info(f"Ready to submit: **{len(current_assessment['questions'])} questions** | **{current_assessment['total_marks']} total marks**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🚀 Submit Assessment", type="primary", help="Finalize and make this assessment available to students"):
                if save_assessments():
                    st.success(f"🎉 Assessment '{assessment_title}' submitted successfully!")
                    st.success(f"✅ Students can now access this assessment with {len(current_assessment['questions'])} questions and {current_assessment['total_marks']} total marks.")
                    st.balloons()
                else:
                    st.error("Failed to submit assessment. Please try again.")
        with col2:
            if st.button("🗑️ Clear Assessment", help="Delete all questions and start over"):
                current_assessment['questions'] = []
                current_assessment['total_marks'] = 0
                save_assessments()
                st.session_state['page'] = "Create Assessment"
                st.session_state['assessment_title'] = assessment_title
                st.rerun()

def attempt_assessment_page():
    """Student view for attempting assessments"""
    st.header("Attempt Assessment")
    
    # Load available assessments
    if not st.session_state.assessments:
        st.warning("NO ASSESSMENTS")
        st.info("No assessments are currently available. Please contact your teacher.")
        return
    
    # Assessment selection
    assessment_titles = list(st.session_state.assessments.keys())
    selected_assessment = st.selectbox("Select Assessment", [""] + assessment_titles)
    
    if not selected_assessment:
        st.info("Please select an assessment to begin.")
        return
    
    assessment = st.session_state.assessments[selected_assessment]
    
    # Display assessment info
    st.subheader(f"Assessment: {assessment['title']}")
    st.info(f"Total Questions: {len(assessment['questions'])} | Total Marks: {assessment['total_marks']}")
    
    # Initialize student answers for this assessment
    assessment_id = assessment['id']
    if assessment_id not in st.session_state.student_answers:
        st.session_state.student_answers[assessment_id] = {}
    
    student_answers = st.session_state.student_answers[assessment_id]
    
    # Display questions with character-by-character input monitoring
    with st.form("assessment_form"):
        st.subheader("Questions")
        
        for i, question in enumerate(assessment['questions']):
            st.write(f"**Question {i+1}** ({question['marks']} marks)")
            st.write(question['text'])
            
            word_limit = question.get('word_limit', 0)
            if word_limit > 0:
                st.info(f"Minimum Word Limit: {word_limit} words")
            
            answer_key = f"q_{question['id']}"
            answer = st.text_area(
                f"Answer {i+1}",
                value=student_answers.get(answer_key, ""),
                key=answer_key,
                placeholder="Type your answer here...",
                height=120
            )
            # Live word count
            word_count = len(answer.strip().split()) if answer.strip() else 0
            st.caption(f"Word Count: {word_count}")
            st.write("---")
        
        submit_assessment = st.form_submit_button("Submit Assessment", type="primary")
    
    if submit_assessment:
        # Check if all questions are answered and meet word limit
        unanswered = []
        below_limit = []
        answers_to_save = {}
        for i, question in enumerate(assessment['questions']):
            answer_key = f"q_{question['id']}"
            answer = st.session_state.get(answer_key, "")
            word_limit = question.get('word_limit', 0)
            if not answer.strip():
                unanswered.append(i+1)
            elif word_limit > 0 and len(answer.strip().split()) < word_limit:
                below_limit.append((i+1, word_limit))
            answers_to_save[answer_key] = answer
        
        if unanswered:
            st.error(f"Please answer all questions. Missing answers for question(s): {', '.join(map(str, unanswered))}")
            return
        if below_limit:
            st.error("The following questions do not meet the minimum word limit: " + ", ".join([f"Q{q} (min {w} words)" for q, w in below_limit]))
            return
        
        # Only now update student_answers
        st.session_state.student_answers[assessment_id] = answers_to_save
        student_answers = answers_to_save
        
        # Grade the assessment
        st.subheader("Grading in Progress...")
        progress_bar = st.progress(0)
        
        total_score = 0
        question_results = []
        
        for i, question in enumerate(assessment['questions']):
            progress_bar.progress((i + 1) / len(assessment['questions']))
            
            answer_key = f"q_{question['id']}"
            student_answer = student_answers.get(answer_key, "")
            
            with st.spinner(f"Grading Question {i+1}..."):
                # Get LLM score
                llm_result = grade_answer(question['text'], question['model_answer'], student_answer, question['marks'])
                
                # Get plagiarism score
                plag_result = check_plagiarism(student_answer)
                
                # Calculate final score
                llm_score = llm_result.get('score', 0)
                llm_feedback = llm_result.get('feedback', 'No feedback available')
                # Remove plagiarism penalty: final_score = llm_score
                final_score = llm_score
                question_results.append({
                    'question_num': i + 1,
                    'question_text': question['text'],
                    'student_answer': student_answer,
                    'max_marks': question['marks'],
                    'llm_score': llm_score,
                    'llm_feedback': llm_feedback,
                    'plagiarism_percentage': plag_result.get('plagiarism_percentage', 0),
                    'plagiarism_penalty': 0,
                    'final_score': final_score
                })
                total_score += final_score
        
        progress_bar.progress(1.0)
        
        # Display results
        st.subheader("Assessment Results")
        
        # Overall score
        percentage = (total_score / assessment['total_marks']) * 100
        st.metric("Total Score", f"{total_score:.1f}/{assessment['total_marks']}")
        
        # Question-wise results
        for result in question_results:
            with st.expander(f"Question {result['question_num']} - Score: {result['final_score']:.1f}/{result['max_marks']}"):
                st.write("**Question:**", result['question_text'])
                st.write("**Your Answer:**", result['student_answer'])
                st.write("**LLM Score:**", f"{result['llm_score']:.1f}/{result['max_marks']}")
                st.write("**AI Feedback:**", result['llm_feedback'])
                st.write(f"**Final Score:** {result['final_score']:.1f}/{result['max_marks']}")
        
        # Overall feedback
        st.subheader("Overall Feedback")
        if percentage >= 80:
            st.success("Excellent work! You have demonstrated strong understanding of the topics.")
        elif percentage >= 60:
            st.info("Good work! There's room for improvement in some areas.")
        else:
            st.warning("Needs improvement. Please review the topics and try again.")

def edit_assessment_page():
    """Show links to previously created assessments that open them in Create Assessment for editing."""
    st.header("Edit Assessment")
    if not st.session_state.assessments:
        st.warning("No assessments available to edit.")
        return
    st.subheader("Previously Created Assessments")
    for title in st.session_state.assessments.keys():
        if st.button(f"Edit '{title}'", key=f"goto_create_{title}"):
            st.session_state['assessment_title'] = title
            st.session_state['nav_override'] = 'Create Assessment'
            st.rerun()

def main():
    """Main application"""
    st.set_page_config(
        page_title="AI Assessment Platform",
        page_icon="🛠️",
        layout="wide"
    )
    
    # Remove the underline from the title
    st.markdown("""
    <style>
    .clickable-title { cursor: pointer; color: #fff; text-decoration: none; border-bottom: none; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="clickable-title">🛠️ Mark 0</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Assessment Platform**")
    
    # Load assessments on startup
    load_assessments()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Navigation override logic
    nav_options = ["Create Assessment", "Edit Assessment", "Attempt Assessment"]
    if 'nav_override' in st.session_state and st.session_state['nav_override']:
        default_index = nav_options.index(st.session_state['nav_override'])
        st.session_state['nav_override'] = None
    elif 'page' in st.session_state and st.session_state['page']:
        default_index = nav_options.index(st.session_state['page'])
    else:
        default_index = 0
    page = st.sidebar.radio(
        "Choose your role:",
        nav_options,
        index=default_index,
        help="Select 'Create Assessment' if you're a teacher, 'Edit Assessment' to modify an existing one, 'Attempt Assessment' if you're a student"
    )
    st.session_state['page'] = page
    # Display selected page
    if page == "Create Assessment":
        create_assessment_page()
    elif page == "Edit Assessment":
        edit_assessment_page()
    else:
        attempt_assessment_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*AI Assessment Platform v1.0*")
    
    # Debug info (can be removed in production)
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.json({
            "Total Assessments": len(st.session_state.assessments),
            "Session State Keys": list(st.session_state.keys())
        })

if __name__ == "__main__":
    main()
