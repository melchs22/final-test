import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from supabase import create_client, Client
import httpx
from httpx_oauth.clients.google import GoogleOAuth2
import os
import asyncio
import plotly.graph_objects as go
import logging
from typing import Optional, Dict, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
METRICS = [
    'attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
    'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume'
]
SESSION_TIMEOUT_MINUTES = 60

# Initialize Supabase client
@st.cache_resource
def get_supabase() -> Client:
    """Initialize and return Supabase client."""
    try:
        return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        st.error("Failed to connect to database. Please try again later.")
        st.stop()
        return None

supabase = get_supabase()

# OAuth setup
def get_google_client() -> GoogleOAuth2:
    """Initialize and return Google OAuth client."""
    try:
        return GoogleOAuth2(
            client_id=st.secrets["oauth"]["client_id"],
            client_secret=st.secrets["oauth"]["client_secret"]
        )
    except Exception as e:
        logger.error(f"Failed to initialize Google OAuth client: {str(e)}")
        st.error("Authentication service unavailable. Please try again later.")
        st.stop()
        return None

async def get_authorization_url(client: GoogleOAuth2) -> str:
    """Get Google OAuth authorization URL."""
    try:
        redirect_uri = st.secrets["oauth"]["redirect_uri"]
        if "STREAMLIT_CLOUD_URL" in os.environ:
            redirect_uri = f"{os.environ['STREAMLIT_CLOUD_URL']}"
        return await client.get_authorization_url(
            redirect_uri=redirect_uri,
            scope=["email", "profile"]
        )
    except Exception as e:
        logger.error(f"Failed to get authorization URL: {str(e)}")
        st.error("Failed to initiate login process. Please try again.")
        st.stop()
        return ""

async def get_google_user(token: Dict) -> Optional[str]:
    """Get Google user email from OAuth token."""
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {token['access_token']}"}
            response = await client.get("https://www.googleapis.com/oauth2/v3/userinfo", headers=headers)
            response.raise_for_status()
            return response.json()['email']
    except Exception as e:
        logger.error(f"Failed to get user info from Google: {str(e)}")
        st.error("Failed to authenticate user. Please try again.")
        return None

# Database Operations with Error Handling
def safe_supabase_operation(operation, *args, **kwargs):
    """Wrapper for Supabase operations with error handling."""
    try:
        result = operation(*args, **kwargs)
        if hasattr(result, 'error') and result.error:
            raise Exception(result.error)
        return result
    except Exception as e:
        logger.error(f"Database operation failed: {str(e)}")
        st.error("Database operation failed. Please try again or contact support.")
        return None

# KPI Functions
@st.cache_data(ttl=300)
def get_kpis() -> Dict[str, float]:
    """Get all KPI thresholds from database."""
    res = safe_supabase_operation(supabase.table("kpis").select("*").execute)
    return {item["metric"]: item["threshold"] for item in res.data} if res else {}

def save_kpis(kpis: Dict[str, float]) -> bool:
    """Save KPI thresholds to database."""
    data = [{"metric": metric, "threshold": threshold} for metric, threshold in kpis.items()]
    res = safe_supabase_operation(supabase.table("kpis").upsert(data).execute)
    return res is not None

# Performance Functions
@st.cache_data(ttl=300)
def get_performance(agent_email: Optional[str] = None) -> pd.DataFrame:
    """Get performance data for specific agent or all agents."""
    query = supabase.table("performance").select("*")
    if agent_email:
        query = query.eq("agent_email", agent_email)
    res = safe_supabase_operation(query.execute)
    return pd.DataFrame(res.data) if res else pd.DataFrame()

def save_performance(agent_email: str, data: Dict) -> bool:
    """Save performance data for an agent."""
    data["agent_email"] = agent_email
    data["date"] = datetime.now().strftime("%Y-%m-%d")
    res = safe_supabase_operation(supabase.table("performance").insert(data).execute)
    return res is not None

# User Functions
@st.cache_data(ttl=3600)
def get_user_role(email: str) -> Optional[str]:
    """Get user role from database."""
    res = safe_supabase_operation(supabase.table("users").select("role").eq("email", email).execute)
    return res.data[0]["role"] if res and res.data else None

@st.cache_data(ttl=3600)
def get_agents() -> List[str]:
    """Get list of all agent emails."""
    res = safe_supabase_operation(supabase.table("users").select("email").eq("role", "Agent").execute)
    return [user["email"] for user in res.data] if res else []

def add_user(email: str, role: str) -> bool:
    """Add a new user to the system."""
    res = safe_supabase_operation(supabase.table("users").insert({"email": email, "role": role}).execute)
    return res is not None

# Goal Functions
@st.cache_data(ttl=300)
def get_goals(agent_email: str) -> pd.DataFrame:
    """Get goals for a specific agent."""
    res = safe_supabase_operation(supabase.table("agent_goals").select("*").eq("agent_email", agent_email).execute)
    return pd.DataFrame(res.data) if res else pd.DataFrame()

def save_goal(agent_email: str, metric: str, goal_value: float) -> bool:
    """Save a new goal for an agent."""
    data = {
        "agent_email": agent_email,
        "metric": metric,
        "goal_value": goal_value,
        "set_date": datetime.now().strftime("%Y-%m-%d")
    }
    res = safe_supabase_operation(supabase.table("agent_goals").insert(data).execute)
    return res is not None

def update_goal(goal_id: int, goal_value: float) -> bool:
    """Update an existing goal."""
    data = {
        "goal_value": goal_value,
        "set_date": datetime.now().strftime("%Y-%m-%d")
    }
    res = safe_supabase_operation(supabase.table("agent_goals").update(data).eq("id", goal_id).execute)
    return res is not None

def delete_goal(goal_id: int) -> bool:
    """Delete a goal."""
    res = safe_supabase_operation(supabase.table("agent_goals").delete().eq("id", goal_id).execute)
    return res is not None

# Feedback Functions
def save_feedback(agent_email: str, feedback: str) -> bool:
    """Save agent feedback."""
    data = {
        "agent_email": agent_email,
        "feedback": feedback,
        "submitted_date": datetime.now().strftime("%Y-%m-%d")
    }
    res = safe_supabase_operation(supabase.table("agent_feedback").insert(data).execute)
    return res is not None

# Training Resources Functions
@st.cache_data(ttl=3600)
def get_resources(metric: str) -> List[Dict]:
    """Get training resources for a specific metric."""
    res = safe_supabase_operation(supabase.table("training_resources").select("*").eq("metric", metric).execute)
    return res.data if res else []

# Achievement Functions
@st.cache_data(ttl=300)
def get_achievements(agent_email: str) -> List[Dict]:
    """Get achievements for a specific agent."""
    res = safe_supabase_operation(supabase.table("agent_achievements").select("*").eq("agent_email", agent_email).execute)
    return res.data if res else []

def award_achievement(agent_email: str, achievement_name: str) -> bool:
    """Award an achievement to an agent if not already awarded."""
    existing = safe_supabase_operation(
        supabase.table("agent_achievements").select("*")
        .eq("agent_email", agent_email)
        .eq("achievement_name", achievement_name)
        .execute
    )
    if not existing or not existing.data:
        data = {
            "agent_email": agent_email,
            "achievement_name": achievement_name,
            "date_earned": datetime.now().strftime("%Y-%m-%d")
        }
        res = safe_supabase_operation(supabase.table("agent_achievements").insert(data).execute)
        return res is not None
    return False

# Preferences Functions
@st.cache_data(ttl=300)
def get_preferences(agent_email: str) -> List[str]:
    """Get dashboard preferences for an agent."""
    res = safe_supabase_operation(supabase.table("agent_preferences").select("preferred_metrics").eq("agent_email", agent_email).execute)
    return res.data[0]['preferred_metrics'] if res and res.data else ['attendance', 'quality_score', 'aht', 'csat']

def save_preferences(agent_email: str, metrics: List[str]) -> bool:
    """Save dashboard preferences for an agent."""
    data = {
        "agent_email": agent_email,
        "preferred_metrics": metrics
    }
    res = safe_supabase_operation(supabase.table("agent_preferences").upsert(data).execute)
    return res is not None

# Performance Assessment
def assess_performance(performance_df: pd.DataFrame, kpis: Dict[str, float]) -> pd.DataFrame:
    """Assess performance against KPIs."""
    results = performance_df.copy()
    for metric in METRICS:
        if metric == 'aht':
            results[f'{metric}_pass'] = results[metric] <= kpis.get(metric, 600)
        else:
            results[f'{metric}_pass'] = results[metric] >= kpis.get(metric, 50)
    results['overall_score'] = results[[f'{m}_pass' for m in METRICS]].mean(axis=1) * 100
    return results

# Session Management
def check_session_timeout():
    """Check if session has timed out and clear if necessary."""
    if 'last_activity' in st.session_state:
        if (datetime.now() - st.session_state.last_activity).seconds > SESSION_TIMEOUT_MINUTES * 60:
            for key in ["user", "role", "oauth_client", "oauth_token"]:
                st.session_state.pop(key, None)
            st.warning("Session expired. Please log in again.")
            st.rerun()
    st.session_state.last_activity = datetime.now()

# Input Validation
def validate_performance_input(data: Dict) -> List[str]:
    """Validate performance input data."""
    errors = []
    for metric, value in data.items():
        if metric.endswith('_percent') and (value < 0 or value > 100):
            errors.append(f"{metric} must be between 0-100")
        elif metric == 'aht' and value < 0:
            errors.append("AHT cannot be negative")
        elif metric == 'call_volume' and value < 0:
            errors.append("Call volume cannot be negative")
    return errors

def validate_goal_input(metric: str, value: float) -> List[str]:
    """Validate goal input data."""
    errors = []
    if metric == 'aht':
        if value < 100 or value > 1000:
            errors.append("AHT target should be between 100-1000 seconds")
    elif metric != 'aht' and (value < 0 or value > 100):
        errors.append("Percentage metrics should be between 0-100")
    elif value == 0:
        errors.append("Target value cannot be zero")
    return errors

# UI Components
def render_login_page():
    """Render the login page."""
    st.title("Login with Google")
    client = get_google_client()
    st.session_state.oauth_client = client

    try:
        auth_url = asyncio.run(get_authorization_url(client))
        st.markdown(f"[Login with Google]({auth_url})")
    except Exception as e:
        st.error(f"Failed to generate login URL: {str(e)}")
        return

    query_params = st.query_params
    if query_params:
        code = query_params.get("code")
        if code:
            try:
                redirect_uri = st.secrets["oauth"]["redirect_uri"]
                if "STREAMLIT_CLOUD_URL" in os.environ:
                    redirect_uri = f"{os.environ['STREAMLIT_CLOUD_URL']}"
                token = asyncio.run(client.get_access_token(code, redirect_uri))
                st.session_state.oauth_token = token
                email = asyncio.run(get_google_user(token))
                if email:
                    role = get_user_role(email)
                    if role:
                        st.session_state.user = email
                        st.session_state.role = role
                        st.success(f"Logged in as {email} ({role})")
                        st.query_params.clear()
                        st.rerun()
                    else:
                        st.error("User not registered in the system. Please contact the administrator.")
            except Exception as e:
                st.error(f"Login failed: {str(e)}. Please try again or contact support.")

def render_manager_dashboard():
    """Render the manager dashboard."""
    st.title("Manager Dashboard")
    tabs = st.tabs(["Set KPIs", "Input Performance", "View Assessments", "Manage Users"])

    with tabs[0]:
        st.header("Set KPI Thresholds")
        kpis = get_kpis()
        with st.form("kpi_form"):
            inputs = {
                'attendance': st.number_input("Attendance (%, min)", value=kpis.get('attendance', 95.0)),
                'quality_score': st.number_input("Quality Score (%, min)", value=kpis.get('quality_score', 90.0)),
                'product_knowledge': st.number_input("Product Knowledge (%, min)", value=kpis.get('product_knowledge', 85.0)),
                'contact_success_rate': st.number_input("Contact Success Rate (%, min)", value=kpis.get('contact_success_rate', 80.0)),
                'onboarding': st.number_input("Onboarding (%, min)", value=kpis.get('onboarding', 90.0)),
                'reporting': st.number_input("Reporting (%, min)", value=kpis.get('reporting', 95.0)),
                'talk_time': st.number_input("CRM Talk Time (seconds, min)", value=kpis.get('talk_time', 300.0)),
                'resolution_rate': st.number_input("Issue Resolution Rate (%, min)", value=kpis.get('resolution_rate', 80.0)),
                'aht': st.number_input("Average Handle Time (seconds, max)", value=kpis.get('aht', 600.0)),
                'csat': st.number_input("Customer Satisfaction (%, min)", value=kpis.get('csat', 85.0)),
                'call_volume': st.number_input("Call Volume (calls, min)", value=kpis.get('call_volume', 50))
            }
            if st.form_submit_button("Save KPIs"):
                save_kpis(inputs)
                st.success("KPIs saved!")

    with tabs[1]:
        st.header("Input Agent Performance")
        agents = get_agents()
        if not agents:
            st.warning("No agents available. Please add agents in the 'Manage Users' tab.")
        else:
            with st.form("performance_form"):
                agent = st.selectbox("Select Agent", agents)
                data = {
                    'attendance': st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'quality_score': st.number_input("Quality Score (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'product_knowledge': st.number_input("Product Knowledge (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'contact_success_rate': st.number_input("Contact Success Rate (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'onboarding': st.number_input("Onboarding (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'reporting': st.number_input("Reporting (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'talk_time': st.number_input("CRM Talk Time (seconds)", min_value=0.0, value=0.0),
                    'resolution_rate': st.number_input("Issue Resolution Rate (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'aht': st.number_input("Average Handle Time (seconds)", min_value=0.0, value=0.0),
                    'csat': st.number_input("Customer Satisfaction (%)", min_value=0.0, max_value=100.0, value=0.0),
                    'call_volume': st.number_input("Call Volume (calls)", min_value=0, value=0)
                }
                if st.form_submit_button("Submit Performance"):
                    errors = validate_performance_input(data)
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        if save_performance(agent, data):
                            st.success("Performance data saved!")
                        else:
                            st.error("Failed to save performance data.")

    with tabs[2]:
        st.header("Assessment Results")
        performance_df = get_performance()
        if not performance_df.empty:
            kpis = get_kpis()
            results = assess_performance(performance_df, kpis)
            st.dataframe(results)
            fig = px.bar(results, x='agent_email', y='overall_score', color='agent_email', title="Agent Overall Scores")
            st.plotly_chart(fig)
        else:
            st.write("No performance data available.")

    with tabs[3]:
        st.header("Manage Users")
        with st.form("user_form"):
            new_email = st.text_input("User Email")
            new_role = st.selectbox("Role", ["Agent", "Manager"])
            if st.form_submit_button("Add User"):
                if not new_email:
                    st.error("Email cannot be empty")
                elif "@" not in new_email:
                    st.error("Please enter a valid email address")
                else:
                    if add_user(new_email, new_role):
                        st.success(f"User {new_email} added as {new_role}!")
                    else:
                        st.error(f"Failed to add user {new_email}")

def render_agent_goals(agent_email: str):
    """Render the agent goals section."""
    st.header("Input Your Goals")
    with st.form("goal_input_form"):
        st.subheader("Add New Goal")
        metric = st.selectbox("Select Metric", METRICS, key="goal_metric")
        goal_value = st.number_input("Target Value", min_value=0.0, max_value=100.0 if metric != 'aht' else 1000.0, value=0.0, step=0.1)
        if st.form_submit_button("Submit Goal"):
            errors = validate_goal_input(metric, goal_value)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                if save_goal(agent_email, metric, goal_value):
                    st.success(f"Goal for {metric.replace('_', ' ').title()} set to {goal_value}!")
                    st.rerun()
                else:
                    st.error("Failed to save goal.")

    st.header("Manage Your Goals")
    goals_df = get_goals(agent_email)
    if not goals_df.empty:
        st.subheader("Your Current Goals")
        goal_data = []
        performance_df = get_performance(agent_email)
        
        for _, goal in goals_df.iterrows():
            goal_id = goal['id']
            metric = goal['metric']
            goal_value = goal['goal_value']
            set_date = goal['set_date']
            latest_value = performance_df[metric].iloc[-1] if not performance_df.empty and metric in performance_df else 0
            progress = (latest_value / goal_value * 100) if metric != 'aht' else (goal_value / latest_value * 100 if latest_value > 0 else 0)
            progress = min(progress, 100)
            goal_data.append({
                "ID": goal_id,
                "Metric": metric.replace('_', ' ').title(),
                "Target Value": goal_value,
                "Current Value": latest_value,
                "Progress (%)": round(progress, 2),
                "Set Date": set_date
            })

        goals_table = pd.DataFrame(goal_data)
        st.dataframe(goals_table)

        st.subheader("Update or Delete Goals")
        for _, goal in goals_df.iterrows():
            goal_id = goal['id']
            metric = goal['metric']
            goal_value = goal['goal_value']
            with st.expander(f"Manage Goal: {metric.replace('_', ' ').title()} (Target: {goal_value})"):
                new_value = st.number_input(
                    f"New Target for {metric}",
                    min_value=0.0,
                    max_value=100.0 if metric != 'aht' else 1000.0,
                    value=float(goal_value),
                    key=f"update_{goal_id}"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Update Goal", key=f"update_btn_{goal_id}"):
                        errors = validate_goal_input(metric, new_value)
                        if errors:
                            for error in errors:
                                st.error(error)
                        else:
                            if update_goal(goal_id, new_value):
                                st.success(f"Updated {metric.replace('_', ' ').title()} goal to {new_value}!")
                                st.rerun()
                            else:
                                st.error("Failed to update goal.")
                with col2:
                    if st.button("Delete Goal", key=f"delete_btn_{goal_id}"):
                        if delete_goal(goal_id):
                            st.success(f"Deleted {metric.replace('_', ' ').title()} goal!")
                            st.rerun()
                        else:
                            st.error("Failed to delete goal.")

        st.subheader("Goal Progress")
        for _, goal in goals_df.iterrows():
            metric = goal['metric']
            goal_value = goal['goal_value']
            latest_value = performance_df[metric].iloc[-1] if not performance_df.empty and metric in performance_df else 0
            progress = (latest_value / goal_value * 100) if metric != 'aht' else (goal_value / latest_value * 100 if latest_value > 0 else 0)
            progress = min(progress, 100)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=progress,
                title={'text': f"Progress to {metric.replace('_', ' ').title()} Goal"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green" if progress >= 80 else "orange"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No goals set yet. Use the 'Input Your Goals' section to add a new goal.")

def render_agent_dashboard(agent_email: str):
    """Render the agent dashboard."""
    st.title(f"Agent Dashboard - {agent_email}")
    performance_df = get_performance(agent_email)
    kpis = get_kpis()

    # Goals Section
    render_agent_goals(agent_email)

    # Metric Breakdown
    st.header("Metric Breakdown")
    selected_metric = st.selectbox("Select Metric to View", METRICS)
    if not performance_df.empty:
        fig = px.line(performance_df, x='date', y=selected_metric, title=f"Your {selected_metric.replace('_', ' ').title()} Over Time")
        kpi_threshold = kpis.get(selected_metric, 50)
        fig.add_hline(y=kpi_threshold, line_dash="dash", line_color="red", annotation_text=f"KPI: {kpi_threshold}")
        st.plotly_chart(fig)
    else:
        st.write("No performance data available for visualization.")

    # Performance Trends
    st.header("Performance Trends")
    if not performance_df.empty:
        trends = {}
        for metric in METRICS:
            last_five = performance_df[metric].tail(5)
            if len(last_five) >= 2:
                trend = "Improving" if last_five.diff().mean() > 0 else "Declining" if last_five.diff().mean() < 0 else "Stable"
                pass_status = "Pass" if (metric == 'aht' and last_five.iloc[-1] <= kpis.get(metric, 600)) or \
                                       (metric != 'aht' and last_five.iloc[-1] >= kpis.get(metric, 50)) else "Fail"
                trends[metric] = {"Trend": trend, "Status": pass_status}
        trends_df = pd.DataFrame(trends).T
        st.dataframe(trends_df.style.applymap(lambda x: 'color: red' if x == 'Fail' else 'color: green', subset=['Status']))

    # Personalized Recommendations
    st.header("Personalized Recommendations")
    recommendations = {
        'aht': "Reduce Average Handle Time by practicing concise communication and using CRM shortcuts.",
        'csat': "Improve Customer Satisfaction by actively listening and personalizing customer interactions.",
        'resolution_rate': "Boost Issue Resolution Rate by reviewing common issues and their solutions in training materials.",
        'attendance': "Ensure consistent attendance by planning your schedule and communicating any issues early.",
        'quality_score': "Enhance Quality Score by double-checking responses for accuracy and clarity.",
        'product_knowledge': "Improve Product Knowledge by reviewing training materials and FAQs regularly."
    }
    if not performance_df.empty:
        latest_performance = performance_df.iloc[-1]
        for metric in METRICS:
            if metric == 'aht' and latest_performance[metric] > kpis.get(metric, 600):
                st.write(f"- {recommendations.get(metric, 'Work on improving this metric.')}")
            elif metric != 'aht' and latest_performance[metric] < kpis.get(metric, 50):
                st.write(f"- {recommendations.get(metric, 'Work on improving this metric.')}")
    else:
        st.write("No recommendations available yet.")

    # Peer Comparison
    st.header("Compare to Team")
    all_performance = get_performance()
    if not all_performance.empty:
        team_avg = all_performance.groupby('date')[METRICS].mean().reset_index().iloc[-1]
        agent_latest = performance_df.iloc[-1] if not performance_df.empty else pd.Series(0, index=METRICS)
        comparison_df = pd.DataFrame({
            'Your Score': agent_latest[METRICS],
            'Team Average': team_avg[METRICS]
        })
        fig = px.bar(comparison_df, barmode='group', title="Your Performance vs. Team Average")
        st.plotly_chart(fig)

    # Feedback Submission
    st.header("Submit Feedback")
    with st.form("feedback_form"):
        feedback = st.text_area("Your Feedback")
        if st.form_submit_button("Submit"):
            if feedback:
                if save_feedback(agent_email, feedback):
                    st.success("Feedback submitted!")
                else:
                    st.error("Failed to submit feedback.")
            else:
                st.warning("Please enter feedback before submitting.")

    # Data Download
    st.header("Download Your Performance Data")
    if not performance_df.empty:
        csv = performance_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"performance_{agent_email}.csv",
            mime="text/csv"
        )

    # Training Resources
    st.header("Training Resources")
    if not performance_df.empty:
        latest_performance = performance_df.iloc[-1]
        for metric in METRICS:
            if metric == 'aht' and latest_performance[metric] > kpis.get(metric, 600):
                resources = get_resources(metric)
                for res in resources:
                    st.markdown(f"- [{res['resource_name']}]({res['resource_url']}) for {metric.replace('_', ' ').title()}")
            elif metric != 'aht' and latest_performance[metric] < kpis.get(metric, 50):
                resources = get_resources(metric)
                for res in resources:
                    st.markdown(f"- [{res['resource_name']}]({res['resource_url']}) for {metric.replace('_', ' ').title()}")

    # Achievements
    st.header("Your Achievements")
    if not performance_df.empty:
        latest_performance = performance_df.iloc[-1]
        if latest_performance['attendance'] >= 100:
            award_achievement(agent_email, "Perfect Attendance")
        if latest_performance['csat'] >= 95:
            award_achievement(agent_email, "CSAT Star")
        if latest_performance['resolution_rate'] >= 90:
            award_achievement(agent_email, "Resolution Master")
        achievements = get_achievements(agent_email)
        for ach in achievements:
            st.write(f"- {ach['achievement_name']} (Earned on {ach['date_earned']})")
    else:
        st.write("No achievements yet.")

    # Dashboard Customization
    st.header("Customize Dashboard")
    preferred_metrics = st.multiselect("Select Metrics to Display", METRICS, default=get_preferences(agent_email))
    if st.button("Save Preferences"):
        if save_preferences(agent_email, preferred_metrics):
            st.success("Preferences saved!")
        else:
            st.error("Failed to save preferences.")

    if not performance_df.empty:
        for metric in preferred_metrics:
            fig = px.line(performance_df, x='date', y=metric, title=f"Your {metric.replace('_', ' ').title()} Over Time")
            st.plotly_chart(fig)

# Main App
def main():
    """Main application function."""
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")

    # Debugging tools
    if st.sidebar.button("Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.oauth_client = None
        st.session_state.oauth_token = None
        st.session_state.last_activity = datetime.now()

    # Check session timeout
    check_session_timeout()

    # Login/Logout
    if not st.session_state.user:
        render_login_page()
        return

    if st.sidebar.button("Logout"):
        for key in ["user", "role", "oauth_client", "oauth_token", "last_activity"]:
            st.session_state.pop(key, None)
        st.rerun()

    # Render appropriate dashboard
    if st.session_state.role == "Manager":
        render_manager_dashboard()
    elif st.session_state.role == "Agent":
        render_agent_dashboard(st.session_state.user)

if __name__ == "__main__":
    main()
