import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json

# Initialize Supabase client
def init_supabase():
    from supabase import create_client
    supabase_url = st.secrets["supabase"]["url"]
    supabase_key = st.secrets["supabase"]["key"]
    return create_client(supabase_url, supabase_key)

# Define KPI Categories and Subcategories for Call Assessments
KPI_CATEGORIES = {
    "Overall Scores on Quality Attributes": [
        "Standard Verbiage & Procedures",
        "Issue Identification",
        "Issue Resolution",
        "Call Courtesy",
        "Call Ticketing And CRM Accuracy"
    ],
    "Greeting and Procedures": [
        "Adherence to Verbiages",
        "NC Reason (Adherence to Verbiages)",
        "Security Checks",
        "NC Reason (Security Checks)"
    ],
    "Issue Identification": [
        "Active Listening",
        "NC Reason (Active Listening)",
        "Effective Probing",
        "NC Reason (Effective Probing)"
    ],
    "Issue Resolution": [
        "Accurate Resolution",
        "NC Reason (Accurate Resolution)",
        "Completeness of Resolution",
        "NC Reason (Completeness of Resolution)"
    ],
    "Call Courtesy": [
        "Politeness & Courtesy",
        "NC Reason (Politeness & Courtesy)",
        "Enthusiasm",
        "NC Reason (Enthusiasm)",
        "Communication Skills",
        "NC Reason (Communication Skills)"
    ],
    "Call Ticketing and Escalation": [
        "CRM Accuracy & Completeness"
    ]
}

# KPI Functions
def get_kpis(supabase):
    try:
        response = supabase.table("kpis").select("*").execute()
        kpis = {}
        for row in response.data:
            for key, value in row.items():
                # Skip non-metric columns like 'id' and 'metric'
                if key in ["id", "metric"] or value is None:
                    continue
                if key == "call_volume":
                    kpis[key] = int(float(value))
                else:
                    kpis[key] = float(value)
        return kpis
    except Exception as e:
        st.error(f"Error retrieving KPIs: {str(e)}")
        return {}

def save_kpis(supabase, kpis):
    try:
        data = {"metric": "kpis"}
        for key, value in kpis.items():
            data[key] = value
        response = supabase.table("kpis").select("*").execute()
        if not response.data:
            supabase.table("kpis").insert(data).execute()
        else:
            supabase.table("kpis").update(data).eq("metric", "kpis").execute()
        return True
    except Exception as e:
        st.error(f"Error saving KPIs: {str(e)}")
        return False

# Performance Functions
def save_performance(supabase, agent_name, data):
    try:
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        performance_data = {
            "agent_name": agent_name,
            "attendance": data.get('attendance', 0.0),
            "quality_score": data.get('quality_score', 0.0),
            "call_volume": int(data.get('call_volume', 0)),
            "resolution_rate": data.get('resolution_rate', 0.0),
            "feedback_score": data.get('feedback_score', 0.0),
            "date": date
        }
        response = supabase.table("performance").select("*").eq("agent_name", agent_name).eq("date", date).execute()
        if not response.data:
            supabase.table("performance").insert(performance_data).execute()
        else:
            supabase.table("performance").update(performance_data).eq("agent_name", agent_name).eq("date", date).execute()
        return True
    except Exception as e:
        st.error(f"Error saving performance: {str(e)}")
        return False

def get_performance(supabase, agent_name=None):
    try:
        if agent_name:
            response = supabase.table("performance").select("*").eq("agent_name", agent_name).execute()
        else:
            response = supabase.table("performance").select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving performance: {str(e)}")
        return pd.DataFrame()

# Feedback Functions
def save_feedback(supabase, agent_name, feedback, manager_name):
    try:
        feedback_data = {
            "agent_name": agent_name,
            "feedback": feedback,
            "manager_name": manager_name,
            "timestamp": datetime.now().isoformat()
        }
        supabase.table("feedback").insert(feedback_data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return False

def get_feedback(supabase, agent_name=None):
    try:
        if agent_name:
            response = supabase.table("feedback").select("*").eq("agent_name", agent_name).execute()
        else:
            response = supabase.table("feedback").select("*").execute()
        if response.data:
            df = pd.DataFrame(response.data)
            # Rename 'created_at' to 'timestamp' if needed
            if 'created_at' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = df['created_at']
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving feedback: {str(e)}")
        return pd.DataFrame()

# Audio Assessment Functions
def upload_audio(supabase, agent_name, audio_file, manager_name):
    try:
        file_name = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_file.name}"
        res = supabase.storage.from_("call-audio").upload(file_name, audio_file.getvalue())
        audio_url = supabase.storage.from_("call-audio").get_public_url(file_name)
        audio_data = {
            "agent_name": agent_name,
            "audio_url": audio_url,
            "upload_timestamp": datetime.now().isoformat(),
            "assessment_notes": {},  # Initialize as empty JSON
            "uploaded_by": manager_name
        }
        supabase.table("audio_assessments").insert(audio_data).execute()
        return True
    except Exception as e:
        st.error(f"Error uploading audio: {str(e)}")
        return False

def get_audio_assessments(supabase, agent_name=None):
    try:
        if agent_name:
            response = supabase.table("audio_assessments").select("*").eq("agent_name", agent_name).execute()
        else:
            response = supabase.table("audio_assessments").select("*").execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving audio assessments: {str(e)}")
        return pd.DataFrame()

def update_assessment_notes(supabase, audio_id, assessment_data):
    try:
        supabase.table("audio_assessments").update({"assessment_notes": assessment_data}).eq("id", audio_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating assessment notes: {str(e)}")
        return False

# Authentication
def get_users(supabase):
    try:
        response = supabase.table("users").select("name").execute()
        if response.data:
            return [user["name"] for user in response.data]
        return []
    except Exception as e:
        st.error(f"Error retrieving users: {str(e)}")
        return []

def authenticate_user(supabase, username, password):
    try:
        response = supabase.table("users").select("*").eq("name", username).eq("password", password).execute()
        if response.data:
            user = response.data[0]
            st.session_state.user = user["name"]
            st.session_state.role = user["role"]
            return True
        return False
    except Exception as e:
        st.error(f"Error authenticating user: {str(e)}")
        return False

# Main App
def main():
    st.set_page_config(page_title="Call Centre Performance Tracker", layout="wide")
    supabase = init_supabase()

    # Initialize session state
    if "user" not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None

    # Authentication
    if not st.session_state.user:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(supabase, username, password):
                st.success(f"Logged in as {st.session_state.user} ({st.session_state.role})")
                st.rerun()
            else:
                st.error("Invalid username or password")
        return

    # Logout Button
    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()

    # Main Interface
    st.title("Call Centre Performance Tracker")
    st.write(f"Logged in as: {st.session_state.user} ({st.session_state.role})")

    tabs = st.tabs([
        "Dashboard",
        "Input Performance",
        "Input Feedback",
        "View Feedback",
        "Set KPI Thresholds",
        "Audio Assessments",
        "Agent Dashboard"
    ])

    # Dashboard
    with tabs[0]:
        st.subheader("Performance Dashboard")
        agents = get_users(supabase)
        selected_agent = st.selectbox("Select Agent", ["All Agents"] + agents, key="dashboard_agent")
        agent_filter = None if selected_agent == "All Agents" else selected_agent
        perf_df = get_performance(supabase, agent_filter)

        if not perf_df.empty:
            perf_df['date'] = pd.to_datetime(perf_df['date'])
            kpis = get_kpis(supabase)
            for metric in ['attendance', 'quality_score', 'call_volume', 'resolution_rate', 'feedback_score']:
                if metric in kpis:
                    fig = px.line(perf_df, x='date', y=metric, color='agent_name',
                                 title=f"{metric.replace('_', ' ').title()} Over Time")
                    fig.add_hline(y=kpis[metric], line_dash="dash", annotation_text=f"Threshold: {kpis[metric]}")
                    st.plotly_chart(fig)
        else:
            st.info("No performance data available.")

    # Input Performance
    with tabs[1]:
        st.subheader("Input Agent Performance")
        if st.session_state.role != "Manager":
            st.warning("Only Managers can input performance data.")
        else:
            agents = [agent for agent in get_users(supabase) if agent != st.session_state.user]
            agent = st.selectbox("Select Agent", agents, key="input_agent")
            date = st.date_input("Select Date", datetime.now())
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 90.0)
            quality_score = st.number_input("Quality Score (%)", 0.0, 100.0, 85.0)
            call_volume = st.number_input("Call Volume", 0, 1000, 50)
            resolution_rate = st.number_input("Resolution Rate (%)", 0.0, 100.0, 80.0)
            feedback_score = st.number_input("Feedback Score (%)", 0.0, 100.0, 90.0)

            if st.button("Submit Performance"):
                data = {
                    "date": date.strftime("%Y-%m-%d"),
                    "attendance": attendance,
                    "quality_score": quality_score,
                    "call_volume": call_volume,
                    "resolution_rate": resolution_rate,
                    "feedback_score": feedback_score
                }
                if save_performance(supabase, agent, data):
                    st.success(f"Performance data saved for {agent} on {date}!")
                else:
                    st.error("Failed to save performance data.")

    # Input Feedback
    with tabs[2]:
        st.subheader("Input Feedback")
        if st.session_state.role != "Manager":
            st.warning("Only Managers can input feedback.")
        else:
            agents = [agent for agent in get_users(supabase) if agent != st.session_state.user]
            agent = st.selectbox("Select Agent", agents, key="feedback_agent")
            feedback = st.text_area("Feedback")
            if st.button("Submit Feedback"):
                if save_feedback(supabase, agent, feedback, st.session_state.user):
                    st.success(f"Feedback saved for {agent}!")
                else:
                    st.error("Failed to save feedback.")

    # View Feedback
    with tabs[3]:
        st.subheader("View Feedback")
        agents = get_users(supabase)
        selected_agent = st.selectbox("Select Agent", ["All Agents"] + agents, key="view_feedback_agent")
        agent_filter = None if selected_agent == "All Agents" else selected_agent
        feedback_df = get_feedback(supabase, agent_filter)

        if not feedback_df.empty:
            if 'timestamp' in feedback_df.columns:
                feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                feedback_df['timestamp'] = "N/A"  # Fallback if timestamp is missing
            st.dataframe(feedback_df)
            st.download_button(label="📥 Download Feedback", data=feedback_df.to_csv(index=False), file_name="feedback.csv")
        else:
            st.info("No feedback available.")

    # Set KPI Thresholds
    with tabs[4]:
        st.subheader("Set KPI Thresholds")
        if st.session_state.role != "Manager":
            st.warning("Only Managers can set KPI thresholds.")
        else:
            kpis = get_kpis(supabase)
            default_kpis = {
                "attendance": 90.0,
                "quality_score": 85.0,
                "call_volume": 50,
                "resolution_rate": 80.0,
                "feedback_score": 90.0
            }
            kpis = {key: kpis.get(key, default) for key, default in default_kpis.items()}

            with st.form("kpi_form"):
                for metric, value in kpis.items():
                    if metric == "call_volume":
                        kpis[metric] = st.number_input(f"{metric.replace('_', ' ').title()}", 0, 1000, int(value))
                    else:
                        kpis[metric] = st.number_input(f"{metric.replace('_', ' ').title()} (%)", 0.0, 100.0, float(value))
                if st.form_submit_button("Save KPI Thresholds"):
                    if save_kpis(supabase, kpis):
                        st.success("KPI thresholds saved!")
                    else:
                        st.error("Failed to save KPI thresholds.")

    # Audio Assessments
    with tabs[5]:
        if st.session_state.role != "Manager":
            st.warning("Only Managers can manage audio assessments.")
        else:
            st.subheader("Upload Audio for Assessment")
            agents = [agent for agent in get_users(supabase) if agent != st.session_state.user]
            agent = st.selectbox("Select Agent", agents, key="audio_agent")
            audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
            if st.button("Upload Audio"):
                if audio_file:
                    if upload_audio(supabase, agent, audio_file, st.session_state.user):
                        st.success(f"Audio uploaded for {agent}!")
                        st.rerun()
                    else:
                        st.error("Failed to upload audio.")
                else:
                    st.error("Please select an audio file.")

            st.subheader("Review Audio Assessments")
            audio_df = get_audio_assessments(supabase)
            if not audio_df.empty:
                audio_df['upload_timestamp'] = pd.to_datetime(audio_df['upload_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                for _, row in audio_df.iterrows():
                    with st.expander(f"{row['agent_name']} - {row['upload_timestamp']}"):
                        st.audio(row['audio_url'], format="audio/mp3")
                        st.write(f"Uploaded by: {row['uploaded_by']}")

                        # Parse existing assessment notes (JSON)
                        assessment_data = row['assessment_notes'] if isinstance(row['assessment_notes'], dict) else {}
                        if isinstance(assessment_data, str):
                            try:
                                assessment_data = json.loads(assessment_data) if assessment_data else {}
                            except:
                                assessment_data = {}

                        # Form to input marks
                        with st.form(key=f"assessment_form_{row['id']}"):
                            st.subheader("Call Assessment Scores (0-10)")
                            new_assessment = {}
                            overall_scores = {}

                            # Iterate through categories and subcategories
                            for category, subcategories in KPI_CATEGORIES.items():
                                st.markdown(f"**{category}**")
                                category_scores = []
                                for subcategory in subcategories:
                                    # Convert subcategory to a key-friendly format
                                    key = subcategory.lower().replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
                                    score = st.number_input(
                                        f"{subcategory} (0-10)",
                                        min_value=0,
                                        max_value=10,
                                        value=int(assessment_data.get(key, 0)),
                                        step=1,
                                        key=f"{key}_{row['id']}"
                                    )
                                    new_assessment[key] = score
                                    category_scores.append(score)

                                # Calculate overall score for the category
                                if category_scores:
                                    overall_score = sum(category_scores) / len(category_scores)
                                    overall_scores[category] = overall_score
                                    st.write(f"Overall {category}: {overall_score:.1f}/10")

                            # Calculate Overall Scores on Quality Attributes
                            quality_attributes = [
                                new_assessment.get("standard_verbiage_and_procedures", 0),
                                new_assessment.get("issue_identification", 0),
                                new_assessment.get("issue_resolution", 0),
                                new_assessment.get("call_courtesy", 0),
                                new_assessment.get("call_ticketing_and_crm_accuracy", 0)
                            ]
                            if quality_attributes:
                                overall_quality_score = sum(quality_attributes) / len(quality_attributes)
                                st.markdown(f"**Overall Scores on Quality Attributes**: {overall_quality_score:.1f}/10")

                            # Submit button
                            if st.form_submit_button("Save Assessment", key=f"save_assessment_{row['id']}"):
                                if update_assessment_notes(supabase, row['id'], new_assessment):
                                    st.success("Assessment saved!")
                                    st.rerun()
                                else:
                                    st.error("Failed to save assessment.")

                # Update the dataframe display to include scores
                display_df = audio_df.copy()
                for category, subcategories in KPI_CATEGORIES.items():
                    for subcategory in subcategories:
                        key = subcategory.lower().replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
                        display_df[subcategory] = display_df['assessment_notes'].apply(
                            lambda x: x.get(key, 0) if isinstance(x, dict) else 0
                        )
                st.dataframe(display_df[['agent_name', 'upload_timestamp', 'uploaded_by'] + 
                                        [subcategory for subcategories in KPI_CATEGORIES.values() for subcategory in subcategories]])
                st.download_button(label="📥 Download Audio Assessments", data=display_df.to_csv(index=False), file_name="audio_assessments.csv")
            else:
                st.info("No audio assessments available.")

    # Agent Dashboard
    with tabs[6]:
        if st.session_state.role != "Agent":
            st.warning("Only Agents can view this dashboard.")
        else:
            st.subheader(f"{st.session_state.user}'s Dashboard")
            
            # Performance Data
            perf_df = get_performance(supabase, st.session_state.user)
            if not perf_df.empty:
                perf_df['date'] = pd.to_datetime(perf_df['date'])
                st.subheader("Performance Metrics")
                for column in ['attendance', 'quality_score', 'call_volume', 'resolution_rate', 'feedback_score']:
                    if column in perf_df.columns:
                        avg_value = perf_df[column].mean()
                        st.metric(label=column.replace('_', ' ').title(), value=f"{avg_value:.1f}")
                fig = px.line(perf_df, x='date', y=['attendance', 'quality_score', 'resolution_rate', 'feedback_score'],
                             title="Performance Over Time", labels={'value': 'Score (%)', 'variable': 'Metric'})
                st.plotly_chart(fig)
            else:
                st.info("No performance data available for you.")

            # Feedback Data
            feedback_df = get_feedback(supabase, st.session_state.user)
            if not feedback_df.empty:
                if 'timestamp' in feedback_df.columns:
                    feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    feedback_df['timestamp'] = "N/A"
                st.subheader("Feedback")
                st.dataframe(feedback_df[['timestamp', 'manager_name', 'feedback']])
            else:
                st.info("No feedback available for you.")

if __name__ == "__main__":
    main()
