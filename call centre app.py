import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client
import requests
import time
import uuid
import urllib.parse
from tenacity import retry, stop_after_attempt, wait_fixed

# SECURITY WARNING: Passwords are stored in plain text in the 'users' table, which is highly insecure.
# This exposes passwords to attackers if the database is compromised.
# Recommendation: Hash passwords using bcrypt or another secure algorithm (e.g., Argon2, PBKDF2).
# Contact the developer to implement password hashing with a migration script for existing passwords.

# Initialize Supabase with caching
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    if not url.startswith("https://"):
        url = f"https://{url}"
    return create_client(url, key)

# Check database tables
@st.cache_data(ttl=3600)
def check_db(_supabase):
    required_tables = ["users", "kpis", "performance", "zoho_agent_data", "goals", "feedback", "notifications", "audio_assessments", "badges", "forum_posts"]
    critical_tables = ["users", "goals", "feedback", "performance"]
    missing_critical = []
    missing_non_critical = []
    
    for table in required_tables:
        try:
            _supabase.table(table).select("count").limit(1).execute()
        except Exception:
            if table in critical_tables:
                missing_critical.append(table)
            else:
                missing_non_critical.append(table)
    
    if missing_critical:
        st.sidebar.error(f"Critical tables missing: {', '.join(missing_critical)}. Please create them.")
        return False
    if missing_non_critical:
        st.sidebar.warning(f"Non-critical tables missing: {', '.join(missing_non_critical)}.")
        if "notifications" in missing_non_critical or "forum_posts" in missing_non_critical:
            st.session_state.notifications_enabled = False
        else:
            st.session_state.notifications_enabled = True
    else:
        st.session_state.notifications_enabled = True
        st.sidebar.success("✅ Connected to database successfully")
    return True

# Save KPIs
def save_kpis(supabase, kpis):
    try:
        for metric, threshold in kpis.items():
            response = supabase.table("kpis").select("*").eq("metric", metric).execute()
            if not response.data:
                supabase.table("kpis").insert({"metric": metric, "threshold": threshold}).execute()
            else:
                supabase.table("kpis").update({"threshold": threshold}).eq("metric", metric).execute()
        return True
    except Exception as e:
        st.error(f"Error saving KPIs: {str(e)}")
        return False

# Get KPIs with caching
@st.cache_data(ttl=3600)
def get_kpis(_supabase):
    try:
        response = _supabase.table("kpis").select("*").execute()
        kpis = {}
        for row in response.data:
            metric = row["metric"]
            value = row["threshold"]
            kpis[metric] = int(float(value)) if metric == "call_volume" else float(value) if value is not None else 0.0
        return kpis
    except Exception as e:
        st.error(f"Error retrieving KPIs: {str(e)}")
        return {}

# Save performance data
def save_performance(supabase, agent_name, data):
    try:
        date = data.get('date', datetime.now().strftime("%Y-%m-%d"))
        performance_data = {
            "agent_name": agent_name,
            "attendance": data['attendance'],
            "quality_score": data['quality_score'],
            "product_knowledge": data['product_knowledge'],
            "contact_success_rate": data['contact_success_rate'],
            "onboarding": data['onboarding'],
            "reporting": data['reporting'],
            "talk_time": data['talk_time'],
            "resolution_rate": data['resolution_rate'],
            "aht": data['aht'],
            "csat": data['csat'],
            "call_volume": data['call_volume'],
            "date": date
        }
        supabase.table("performance").insert(performance_data).execute()
        kpis = get_kpis(supabase)
        for metric, value in performance_data.items():
            if metric in kpis and metric not in ["agent_name", "date"]:
                threshold = kpis[metric]
                badge_name = f"{metric.replace('_', ' ').title()} Star"
                if (metric == "aht" and value <= threshold * 0.9) or (metric != "aht" and value >= threshold * 1.1):
                    description = f"Achieved exceptional {metric.replace('_', ' ')} of {value:.1f}{' sec' if metric == 'aht' else '%'}"
                    award_badge(supabase, agent_name, badge_name, description, "System")
                if (metric == "aht" and value <= threshold * 0.9) or (metric != "aht" and value >= threshold * 1.1):
                    send_performance_alert(supabase, agent_name, metric, value, threshold, is_positive=True)
                elif (metric == "aht" and value > threshold * 1.1) or (metric != "aht" and value < threshold * 0.9):
                    send_performance_alert(supabase, agent_name, metric, value, threshold, is_positive=False)
        update_goal_status(supabase, agent_name)
        return True
    except Exception as e:
        st.error(f"Error saving performance data: {str(e)}")
        return False

# Get performance data with caching
@st.cache_data(ttl=600)
def get_performance(_supabase, agent_name=None):
    try:
        query = _supabase.table("performance").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            numeric_cols = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
                           'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'call_volume' in df.columns:
                df['call_volume'] = pd.to_numeric(df['call_volume'], errors='coerce').fillna(0).astype(int)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving performance data: {str(e)}")
        return pd.DataFrame()

# Get Zoho agent data with caching
@st.cache_data(ttl=3600)
def get_zoho_agent_data(_supabase, agent_name=None):
    try:
        all_data = []
        chunk_size = 1000
        offset = 0
        while True:
            query = _supabase.table("zoho_agent_data").select("*").range(offset, offset + chunk_size - 1)
            if agent_name:
                query = query.eq("ticket_owner", agent_name)
            response = query.execute()
            if not response.data:
                break
            all_data.extend(response.data)
            if len(response.data) < chunk_size:
                break
            offset += chunk_size
        if all_data:
            df = pd.DataFrame(all_data)
            if 'id' not in df.columns or 'ticket_owner' not in df.columns:
                st.error("Zoho table missing required columns (id, ticket_owner).")
                return pd.DataFrame()
            return df
        st.warning("No Zoho data found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving Zoho data: {str(e)}")
        return pd.DataFrame()

# Set agent goal
def set_agent_goal(supabase, agent_name, metric, target_value, manager_name, is_manager=False):
    try:
        goal_data = {
            "agent_name": agent_name,
            "metric": metric,
            "target_value": target_value,
            "status": "Approved" if is_manager else "Awaiting Approval",
            "created_at": datetime.now().isoformat()
        }
        response = supabase.table("goals").select("*").eq("agent_name", agent_name).eq("metric", metric).execute()
        if response.data:
            supabase.table("goals").update(goal_data).eq("agent_name", agent_name).eq("metric", metric).execute()
        else:
            supabase.table("goals").insert(goal_data).execute()
        return True
    except Exception as e:
        st.error(f"Error setting goal: {str(e)}")
        return False

# Approve or reject goal
def approve_goal(supabase, goal_id, manager_name, approve=True):
    try:
        update_data = {
            "status": "Approved" if approve else "Rejected",
            "approved_at": datetime.now().isoformat()
        }
        supabase.table("goals").update(update_data).eq("id", goal_id).execute()
        if st.session_state.get("notifications_enabled", False):
            goal = supabase.table("goals").select("agent_name").eq("id", goal_id).execute()
            if goal.data:
                agent_name = goal.data[0]["agent_name"]
                agent = supabase.table("users").select("id").eq("name", agent_name).execute()
                if agent.data:
                    status = "approved" if approve else "rejected"
                    supabase.table("notifications").insert({
                        "user_id": agent.data[0]["id"],
                        "message": f"Your goal for {agent_name} was {status} by {manager_name}"
                    }).execute()
        return True
    except Exception as e:
        st.error(f"Error approving/rejecting goal: {str(e)}")
        return False

# Update goal status
def update_goal_status(supabase, agent_name):
    try:
        goals = supabase.table("goals").select("*").eq("agent_name", agent_name).in_("status", ["Approved", "Pending"]).execute()
        if not goals.data:
            return
        perf = get_performance(supabase, agent_name)
        if perf.empty:
            return
        perf['date'] = pd.to_datetime(perf['date'], errors='coerce')
        if perf['date'].isna().all():
            return
        latest_perf = perf[perf['date'] == perf['date'].max()]
        if latest_perf.empty:
            return
        for goal in goals.data:
            metric = goal['metric']
            target = float(goal['target_value'])
            if metric in latest_perf.columns:
                value = float(latest_perf[metric].iloc[0])
                if (metric == "aht" and value <= target) or (metric != "aht" and value >= target):
                    status = "Completed"
                    badge_name = f"{metric.replace('_', ' ').title()} Master"
                    description = f"Achieved {metric.replace('_', ' ')} goal of {target:.1f}{' sec' if metric == 'aht' else '%'} with {value:.1f}{' sec' if metric == 'aht' else '%'}"
                    award_badge(supabase, agent_name, badge_name, description, "System")
                else:
                    status = goal['status']
                supabase.table("goals").update({"status": status}).eq("id", goal['id']).execute()
    except Exception as e:
        st.error(f"Error updating goal status: {str(e)}")

# Get feedback with caching
@st.cache_data(ttl=600)
def get_feedback(_supabase, agent_name=None):
    try:
        query = _supabase.table("feedback").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        st.warning("No feedback found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving feedback: {str(e)}")
        return pd.DataFrame()

# Respond to feedback
def respond_to_feedback(supabase, feedback_id, manager_response, manager_name):
    try:
        response_data = {
            "manager_response": manager_response,
            "response_timestamp": datetime.now().isoformat()
        }
        supabase.table("feedback").update(response_data).eq("id", feedback_id).execute()
        if st.session_state.get("notifications_enabled", False):
            feedback = supabase.table("feedback").select("agent_name").eq("id", feedback_id).execute()
            if feedback.data:
                agent_name = feedback.data[0]["agent_name"]
                agent = supabase.table("users").select("id").eq("name", agent_name).execute()
                if agent.data:
                    supabase.table("notifications").insert({
                        "user_id": agent.data[0]["id"],
                        "message": f"Manager responded to your feedback: {manager_response[:50]}..."
                    }).execute()
        return True
    except Exception as e:
        st.error(f"Error responding to feedback: {str(e)}")
        return False

# Get notifications with caching
@st.cache_data(ttl=300)
def get_notifications(_supabase):
    if not st.session_state.get("notifications_enabled", False):
        return pd.DataFrame()
    try:
        user_response = _supabase.table("users").select("id").eq("name", st.session_state.user).execute()
        if not user_response.data:
            return pd.DataFrame()
        user_id = user_response.data[0]["id"]
        response = _supabase.table("notifications").select("*").eq("user_id", user_id).eq("read", False).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving notifications: {str(e)}")
        return pd.DataFrame()

# Send performance alert
def send_performance_alert(supabase, agent_name, metric, value, threshold, is_positive=True):
    try:
        agent = supabase.table("users").select("id").eq("name", agent_name).execute()
        if agent.data:
            message = f"{'Great job' if is_positive else 'Attention'}: {metric.replace('_', ' ').title()} {'exceeded' if is_positive else 'below'} {threshold:.1f}{' sec' if metric == 'aht' else '%'} with {value:.1f}{' sec' if metric == 'aht' else '%'}"
            supabase.table("notifications").insert({
                "user_id": agent.data[0]["id"],
                "message": message
            }).execute()
        return True
    except Exception as e:
        st.error(f"Error sending alert: {str(e)}")
        return False

# Award badge
def award_badge(supabase, agent_name, badge_name, description, awarded_by):
    try:
        existing = supabase.table("badges").select("id").eq("agent_name", agent_name).eq("badge_name", badge_name).execute()
        if existing.data:
            return False
        supabase.table("badges").insert({
            "agent_name": agent_name,
            "badge_name": badge_name,
            "description": description,
            "awarded_by": awarded_by,
            "earned_at": datetime.now().isoformat()
        }).execute()
        if st.session_state.get("notifications_enabled", False):
            agent = supabase.table("users").select("id").eq("name", agent_name).execute()
            if agent.data:
                supabase.table("notifications").insert({
                    "user_id": agent.data[0]["id"],
                    "message": f"You earned the '{badge_name}' badge: {description}"
                }).execute()
        return True
    except Exception as e:
        st.error(f"Error awarding badge: {str(e)}")
        return False

# Get leaderboard with caching
@st.cache_data(ttl=3600)
def get_leaderboard(_supabase):
    try:
        response = _supabase.table("performance").select("agent_name").execute()
        if response.data:
            df_perf = pd.DataFrame(response.data)
            all_perf_response = _supabase.table("performance").select("*").execute()
            if not all_perf_response.data:
                return pd.DataFrame()
            df_all = pd.DataFrame(all_perf_response.data)
            kpis = get_kpis(_supabase)
            results = assess_performance(df_all, kpis)
            leaderboard_df = results.groupby("agent_name")["overall_score"].mean().reset_index()
            badges_response = _supabase.table("badges").select("agent_name, id").execute()
            badges_df = pd.DataFrame(badges_response.data) if badges_response.data else pd.DataFrame(columns=["agent_name", "id"])
            badge_counts = badges_df.groupby("agent_name")["id"].nunique().reset_index(name="badges_earned")
            leaderboard_df = leaderboard_df.merge(badge_counts, on="agent_name", how="left").fillna({"badges_earned": 0})
            leaderboard_df["badges_earned"] = leaderboard_df["badges_earned"].astype(int)
            leaderboard_df = leaderboard_df.sort_values("overall_score", ascending=False)
            return leaderboard_df
    except Exception as e:
        st.error(f"Unable to retrieve leaderboard: {str(e)}")
        return pd.DataFrame()

# Create forum post
def create_forum_post(supabase, user_name, message, category):
    try:
        supabase.table("forum_posts").insert({
            "user_name": user_name,
            "message": message,
            "category": category,
            "created_at": datetime.now().isoformat()
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error creating forum post: {str(e)}")
        return False

# Get forum posts with caching
@st.cache_data(ttl=600)
def get_forum_posts(_supabase, category=None):
    try:
        query = _supabase.table("forum_posts").select("*")
        if category:
            query = query.eq("category", category)
        response = query.order("created_at", desc=True).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            badge_counts = _supabase.table("badges").select("agent_name, count(id)").group("agent_name").execute()
            badge_dict = {row['agent_name']: row['count'] for row in badge_counts.data} if badge_counts.data else {}
            df['badge_count'] = df['user_name'].map(badge_dict).fillna(0).astype(int)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving forum posts: {str(e)}")
        return pd.DataFrame()

# Get AI coaching tips
@st.cache_data(ttl=3600)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_coaching_tips(_supabase, agent_name):
    try:
        perf = get_performance(_supabase, agent_name)
        if perf.empty:
            return []
        latest_perf = perf[perf['date'] == perf['date'].max()]
        kpis = get_kpis(_supabase)
        tips = []
        api_token = st.secrets.get("huggingface", {}).get("api_token", None)
        if not api_token:
            st.warning("Hugging Face API token not found.")
            return [{"metric": "general", "tip": "Focus on improving overall performance."}]
        for metric in ['attendance', 'quality_score', 'csat', 'aht']:
            if metric in latest_perf.columns:
                value = float(latest_perf[metric].iloc[0])
                threshold = kpis.get(metric, 600 if metric == 'aht' else 50)
                if (metric == "aht" and value > threshold) or (metric != "aht" and value < threshold):
                    prompt = f"You are a call center coach. Provide a concise, actionable coaching tip for an agent whose {metric.replace('_', ' ')} is {value:.1f}{' sec' if metric == 'aht' else '%'}, below the target of {threshold:.1f}{' sec' if metric == 'aht' else '%'}."
                    headers = {"Authorization": f"Bearer {api_token}"}
                    response = requests.post(
                        "https://api-inference.huggingface.co/models/google/flan-t5-small",
                        headers=headers,
                        json={"inputs": prompt, "parameters": {"max_length": 50}},
                        timeout=15
                    )
                    response.raise_for_status()
                    data = response.json()
                    tip = data[0]['generated_text'].strip() if isinstance(data, list) and data else f"Focus on improving {metric.replace('_', ' ')}."
                    tips.append({"metric": metric, "tip": tip})
        return tips if tips else [{"metric": "general", "tip": "You're doing well! Keep it up."}]
    except Exception as e:
        st.error(f"Error generating coaching tips: {str(e)}")
        return [{"metric": "general", "tip": "Please consult your manager for coaching tips."}]

# Ask the AI coach
@st.cache_data(ttl=3600)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def ask_coach(_supabase, agent_name, question):
    try:
        perf = get_performance(_supabase, agent_name)
        context = ""
        if not perf.empty:
            latest_perf = perf[perf['date'] == perf['date'].max()]
            metrics = ['attendance', 'quality_score', 'csat', 'aht']
            context = "Agent's latest performance: " + ", ".join(
                f"{m.replace('_', ' ')}: {float(latest_perf[m].iloc[0]):.1f}{' sec' if m == 'aht' else '%'}"
                for m in metrics if m in latest_perf.columns
            ) + ". "
        api_token = st.secrets.get("huggingface", {}).get("api_token", None)
        if not api_token:
            st.warning("Hugging Face API token not found.")
            return "Please consult your manager."
        prompt = f"You are a call center coach. {context}Answer the agent's question concisely: {question}"
        headers = {"Authorization": f"Bearer {api_token}"}
        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-small",
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_length": 100}},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        answer = data[0]['generated_text'].strip() if isinstance(data, list) and data else "Please consult your manager."
        return answer
    except Exception as e:
        st.error(f"Error generating coach response: {str(e)}")
        return "Please consult your manager for assistance."

# Plot interactive performance chart
@st.cache_data(ttl=600)
def plot_performance_chart(_supabase, agent_name=None, metrics=None):
    try:
        df = get_performance(_supabase, agent_name)
        if df.empty:
            return None
        if metrics is None:
            metrics = ['attendance', 'quality_score', 'csat', 'resolution_rate']
        if agent_name:
            latest_df = df[df['date'] == df['date'].max()]
            values = [latest_df[m].mean() for m in metrics]
            fig = go.Figure(data=go.Scatterpolar(r=values, theta=[m.replace('_', ' ').title() for m in metrics], fill='toself'))
            fig.update_layout(title=f"Performance Profile for {agent_name}", polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        else:
            avg_df = df.groupby('agent_name')[metrics].mean().reset_index()
            fig = px.bar(avg_df, x='agent_name', y=metrics, barmode='group', title="Team Performance Comparison")
            fig.update_layout(yaxis_title="Value (%)", xaxis_title="Agent")
        return fig
    except Exception as e:
        st.error(f"Error plotting chart: {str(e)}")
        return None

# Assess performance
def assess_performance(performance_df, kpis):
    if performance_df.empty:
        return performance_df
    results = performance_df.copy()
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'csat', 'call_volume']
    for metric in metrics:
        if metric in results.columns:
            results[f'{metric}_pass'] = results[metric] <= kpis.get(metric, 600) if metric == 'aht' else results[metric] >= kpis.get(metric, 50)
    pass_columns = [f'{m}_pass' for m in metrics if f'{m}_pass' in results.columns]
    if pass_columns:
        results['overall_score'] = results[pass_columns].mean(axis=1) * 100
    return results

# Authenticate user (plain text passwords)
@st.cache_data(ttl=3600)
def authenticate_user(_supabase, name, password):
    try:
        user_response = _supabase.table("users").select("*").eq("name", name).execute()
        if user_response.data:
            stored_password = user_response.data[0]["password"]
            if password == stored_password:
                return True, name, user_response.data[0]["role"]
        return False, None, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None, None

# Change password (plain text)
def change_password(supabase, username, current_password, new_password, confirm_password):
    try:
        # Validate inputs
        if new_password != confirm_password:
            return False, "New password and confirmation do not match."
        if len(new_password) < 8:
            return False, "New password must be at least 8 characters long."
        user_response = supabase.table("users").select("*").eq("name", username).execute()
        if not user_response.data:
            return False, "User not found."
        stored_password = user_response.data[0]["password"]
        if current_password != stored_password:
            return False, "Current password is incorrect."
        supabase.table("users").update({"password": new_password}).eq("name", username).execute()
        return True, "Password changed successfully!"
    except Exception as e:
        return False, f"Error changing password: {str(e)}"

# Setup real-time polling
def setup_realtime(supabase):
    if st.session_state.get("auto_refresh", False):
        current_time = datetime.now()
        last_refresh = st.session_state.get("last_refresh", current_time)
        if current_time - last_refresh >= timedelta(seconds=30):
            st.session_state.data_updated = True
            st.session_state.last_refresh = current_time
            get_performance.clear()
            get_feedback.clear()
            get_notifications.clear()
            get_forum_posts.clear()
            plot_performance_chart.clear()
        st.sidebar.success("Auto-refresh enabled.")

# Upload audio
def upload_audio(supabase, agent_name, audio_file, manager_name):
    try:
        file_name = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_file.name}"
        res = supabase.storage.from_("call-audio").upload(file_name, audio_file.getvalue())
        audio_url = supabase.storage.from_("call-audio").get_public_url(file_name)
        supabase.table("audio_assessments").insert({
            "agent_name": agent_name,
            "audio_url": audio_url,
            "upload_timestamp": datetime.now().isoformat(),
            "assessment_notes": "",
            "uploaded_by": manager_name
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error uploading audio: {str(e)}")
        return False

# Get audio assessments with caching
@st.cache_data(ttl=600)
def get_audio_assessments(_supabase, agent_name=None):
    try:
        query = _supabase.table("audio_assessments").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        st.warning("No audio assessments found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving audio assessments: {str(e)}")
        return pd.DataFrame()

# Update assessment notes
def update_assessment_notes(supabase, audio_id, notes):
    try:
        supabase.table("audio_assessments").update({"assessment_notes": notes}).eq("id", audio_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating assessment notes: {str(e)}")
        return False

# Main application
def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    
    # Personalized Dashboard Themes
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"
    theme_css = {
        "light": """
            .reportview-container { background: linear-gradient(to right, #f0f4f8, #e0e7ff); --card-bg: #ffffff; --text-color: #2c3e50; --metric-color: #4CAF50; --button-bg: #4CAF50; --button-hover: #388E3C; }
            .sidebar .sidebar-content { background-color: #ffffff; border-right: 2px solid #4CAF50; }
            .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 8px 16px; }
            .stButton>button:hover { background-color: #388E3C; }
            h1, h2, h3 { color: #2c3e50; font-family: 'Arial', sans-serif; }
            .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .progress-bar { height: 20px; border-radius: 5px; }
            .feedback-container { max-height: 400px; overflow-y: auto; padding: 10px; background-color: #e5ddd5; border-radius: 8px; }
            .feedback-item { margin: 10px 0; padding: 10px; border-radius: 10px; max-width: 70%; }
            .agent-msg { background-color: #dcf8c6; margin-left: auto; text-align: right; }
            .manager-msg { background-color: #fff; margin-right: auto; }
            .timestamp { font-size: 0.7em; color: #666; }
            iframe { border: 2px solid #1f77b4; border-radius: 10px; background: white; }
        """,
        "dark": """
            .reportview-container { background: linear-gradient(to right, #2c3e50, #34495e); color: white; --card-bg: #34495e; --text-color: #ecf0f1; --metric-color: #3498db; --button-bg: #3498db; --button-hover: #2980b9; }
            .sidebar .sidebar-content { background-color: #34495e; border-right: 2px solid #3498db; }
            .stButton>button { background-color: #3498db; color: white; }
            .stButton>button:hover { background-color: #2980b9; }
            h1, h2, h3 { color: #ecf0f1; }
            .stMetric { background-color: #34495e; color: white; }
            .progress-bar { height: 20px; border-radius: 5px; }
            .feedback-container { max-height: 400px; overflow-y: auto; padding: 10px; background-color: #2c3e50; border-radius: 8px; }
            .feedback-item { margin: 10px 0; padding: 10px; border-radius: 10px; max-width: 70%; }
            .agent-msg { background-color: #3498db; margin-left: auto; text-align: right; color: white; }
            .manager-msg { background-color: #ecf0f1; margin-right: auto; color: black; }
            .timestamp { font-size: 0.7em; color: #bdc3c7; }
            iframe { border: 2px solid #3498db; border-radius: 10px; background: white; }
        """,
        "blue": """
            .reportview-container { background: linear-gradient(to right, #3498db, #2980b9); color: white; --card-bg: #2980b9; --text-color: #ecf0f1; --metric-color: #2ecc71; --button-bg: #2ecc71; --button-hover: #27ae60; }
            .sidebar .sidebar-content { background-color: #2980b9; border-right: 2px solid #3498db; }
            .stButton>button { background-color: #2ecc71; color: white; }
            .stButton>button:hover { background-color: #27ae60; }
            h1, h2, h3 { color: #ecf0f1; }
            .stMetric { background-color: #2980b9; color: white; }
            .progress-bar { height: 20px; border-radius: 5px; }
            .feedback-container { max-height: 400px; overflow-y: auto; padding: 10px; background-color: #3498db; border-radius: 8px; }
            .feedback-item { margin: 10px 0; padding: 10px; border-radius: 10px; max-width: 70%; }
            .agent-msg { background-color: #2ecc71; margin-left: auto; text-align: right; color: white; }
            .manager-msg { background-color: #ecf0f1; margin-right: auto; color: black; }
            .timestamp { font-size: 0.7em; color: #bdc3c7; }
            iframe { border: 2px solid #2ecc71; border-radius: 10px; background: white; }
        """
    }
    
    # Custom CSS for UI components
    custom_css = """
        /* From Uiverse.io by Na3ar-17 */
        .container {
            padding: 0;
            margin: 0;
            box-sizing: border-box;
            font-family: Arial, Helvetica, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .label {
            background-color: transparent;
            border: 2px solid rgb(91, 91, 240);
            display: flex;
            align-items: center;
            border-radius: 50px;
            width: 160px;
            cursor: pointer;
            transition: all 0.4s ease;
            padding: 5px;
            position: relative;
        }
        .label::before {
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #fff;
            width: 8px;
            height: 8px;
            transition: all 0.4s ease;
            border-radius: 100%;
            margin: auto;
            opacity: 0;
            visibility: hidden;
        }
        .label .input {
            display: none;
        }
        .label .title {
            font-size: 17px;
            color: #fff;
            transition: all 0.4s ease;
            position: absolute;
            right: 18px;
            bottom: 14px;
            text-align: center;
        }
        .label .title:last-child {
            opacity: 0;
            visibility: hidden;
        }
        .label .circle {
            height: 45px;
            width: 45px;
            border-radius: 50%;
            background-color: rgb(91, 91, 240);
            display: flex;
            justify-content: center;
            align-items: center;
            transition: all 0.4s ease;
            position: relative;
            box-shadow: 0 0 0 0 rgb(255, 255, 255);
            overflow: hidden;
        }
        .label .circle .icon {
            color: #fff;
            width: 30px;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.4s ease;
        }
        .label .circle .square {
            aspect-ratio: 1;
            width: 15px;
            border-radius: 2px;
            background-color: #fff;
            opacity: 0;
            visibility: hidden;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.4s ease;
        }
        .label .circle::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            background-color: #3333a8;
            width: 100%;
            height: 0;
            transition: all 0.4s ease;
        }
        .label:has(.input:checked) {
            width: 57px;
            animation: installed 0.4s ease 3.5s forwards;
        }
        .label:has(.input:checked)::before {
            animation: rotate 3s ease-in-out 0.4s forwards;
        }
        .label .input:checked + .circle {
            animation:
                pulse 1s forwards,
                circleDelete 0.2s ease 3.5s forwards;
            rotate: 180deg;
        }
        .label .input:checked + .circle::before {
            animation: installing 3s ease-in-out forwards;
        }
        .label .input:checked + .circle .icon {
            opacity: 0;
            visibility: hidden;
        }
        .label .input:checked ~ .circle .square {
            opacity: 1;
            visibility: visible;
        }
        .label .input:checked ~ .title {
            opacity: 0;
            visibility: hidden;
        }
        .label .input:checked ~ .title:last-child {
            animation: showInstalledMessage 0.4s ease 3.5s forwards;
        }
        @keyframes pulse {
            0% { scale: 0.95; box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7); }
            70% { scale: 1; box-shadow: 0 0 0 16px rgba(255, 255, 255, 0); }
            100% { scale: 0.95; box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
        }
        @keyframes installing {
            from { height: 0; }
            to { height: 100%; }
        }
        @keyframes rotate {
            0% { transform: rotate(-90deg) translate(27px) rotate(0); opacity: 1; visibility: visible; }
            99% { transform: rotate(270deg) translate(27px) rotate(270deg); opacity: 1; visibility: visible; }
            100% { opacity: 0; visibility: hidden; }
        }
        @keyframes installed {
            100% { width: 150px; border-color: rgb(35, 174, 35); }
        }
        @keyframes circleDelete {
            100% { opacity: 0; visibility: hidden; }
        }
        @keyframes showInstalledMessage {
            100% { opacity: 1; visibility: visible; right: 56px; }
        }
        /* From Uiverse.io by JkHuger */
        .login {
            color: #000;
            text-transform: uppercase;
            letter-spacing: 2px;
            display: block;
            font-weight: bold;
            font-size: x-large;
            text-align: center;
            margin-bottom: 20px;
        }
        .card {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 350px;
            width: 300px;
            flex-direction: column;
            gap: 35px;
            background: #e3e3e3;
            box-shadow: 16px 16px 32px #c8c8c8, -16px -16px 32px #fefefe;
            border-radius: 8px;
            margin: 50px auto;
        }
        .inputBox {
            position: relative;
            width: 250px;
        }
        .inputBox input {
            width: 100%;
            padding: 10px;
            outline: none;
            border: none;
            color: #000;
            font-size: 1em;
            background: transparent;
            border-left: 2px solid #000;
            border-bottom: 2px solid #000;
            transition: 0.1s;
            border-bottom-left-radius: 8px;
        }
        .inputBox span {
            margin-top: 5px;
            position: absolute;
            left: 0;
            transform: translateY(-4px);
            margin-left: 10px;
            padding: 10px;
            pointer-events: none;
            font-size: 12px;
            color: #000;
            text-transform: uppercase;
            transition: 0.5s;
            letter-spacing: 3px;
            border-radius: 8px;
        }
        .inputBox input:valid~span,
        .inputBox input:focus~span {
            transform: translateX(113px) translateY(-15px);
            font-size: 0.8em;
            padding: 5px 10px;
            background: #000;
            letter-spacing: 0.2em;
            color: #fff;
            border: 2px;
        }
        .inputBox input:valid,
        .inputBox input:focus {
            border: 2px solid #000;
            border-radius: 8px;
        }
        .enter {
            height: 45px;
            width: 100px;
            border-radius: 5px;
            border: 2px solid #000;
            cursor: pointer;
            background-color: transparent;
            transition: 0.5s;
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 2px;
            margin-bottom: 1em;
        }
        .enter:hover {
            background-color: rgb(0, 0, 0);
            color: white;
        }
    """
    st.markdown(f"<style>{theme_css[st.session_state.theme]}{custom_css}</style>", unsafe_allow_html=True)

    # Initialize Supabase
    try:
        supabase = init_supabase()
        if not check_db(supabase):
            st.error("Critical database tables are missing.")
            st.stop()
        st.session_state.supabase = supabase
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        st.stop()

    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.data_updated = False
        st.session_state.notifications_enabled = False
        st.session_state.auto_refresh = False
        st.session_state.last_refresh = datetime.now()
        st.session_state.cleared_chats = set()
        st.session_state.theme = "light"
        st.session_state.theme_index = 0

    # Theme toggle
    themes = ["light", "dark", "blue"]
    def toggle_theme():
        st.session_state.theme_index = (st.session_state.theme_index + 1) % len(themes)
        st.session_state.theme = themes[st.session_state.theme_index]
        st.rerun()

    # Login
    if not st.session_state.user:
        st.markdown('<h1 class="login">🔐 Login</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown('<div class="inputBox"><input type="text" required="required" id="name"><span>Name</span></div>', unsafe_allow_html=True)
            name = st.text_input("", key="login_name", label_visibility="collapsed")
            st.markdown('<div class="inputBox"><input type="password" required="required" id="password"><span>Password</span></div>', unsafe_allow_html=True)
            password = st.text_input("", type="password", key="login_password", label_visibility="collapsed")
            st.markdown('<button class="enter" type="submit">Enter</button>', unsafe_allow_html=True)
            if st.form_submit_button("Login", use_container_width=True):
                success, user, role = authenticate_user(supabase, name, password)
                if success:
                    st.session_state.user = user
                    st.session_state.role = role
                    st.success(f"Logged in as {user} ({role})")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="container">', unsafe_allow_html=True)
        st.markdown(f'''
            <label class="label">
                <input class="input" type="checkbox" {'checked' if st.session_state.theme != 'light' else ''} onclick="window.location.reload()">
                <span class="circle">
                    <img class="icon" src="https://cdn-icons-png.flaticon.com/512/1160/1160119.png"/>
                    <span class="square"></span>
                </span>
                <span class="title">{st.session_state.theme.title()}</span>
                <span class="title">Change Theme</span>
            </label>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Toggle Theme", key="theme_toggle"):
            toggle_theme()
        st.info(f"👤 Logged in as: {st.session_state.user}")
        st.info(f"🎓 Role: {st.session_state.role}")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.role = None
            st.rerun()
        st.session_state.auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.get("auto_refresh", False))
        setup_realtime(supabase)

    # Notifications
    if st.session_state.get("notifications_enabled", False):
        notifications = get_notifications(supabase)
        with st.sidebar.expander(f"🔔 Notifications ({len(notifications)})"):
            if notifications.empty:
                st.write("No new notifications.")
            else:
                for _, notif in notifications.iterrows():
                    st.markdown(f"<div style='color: {'green' if 'Great job' in notif['message'] else 'red' if 'Attention' in notif['message'] else 'black'}'>{notif['message']}</div>", unsafe_allow_html=True)
                    if st.button("Mark as Read", key=f"notif_{notif['id']}"):
                        supabase.table("notifications").update({"read": True}).eq("id", notif["id"]).execute()
                        get_notifications.clear()
                        st.rerun()
    else:
        with st.sidebar.expander("🔔 Notifications (0)"):
            st.write("Notifications disabled.")

    # Display company logo
    try:
        st.image(r"./companylogo.png", width=150)
    except Exception:
        st.warning("Failed to load company logo.")

    # Manager Dashboard
    if st.session_state.role == "Manager":
        st.title("📊 Manager Dashboard")
        performance_df = get_performance(supabase)
        if not performance_df.empty:
            kpis = get_kpis(supabase)
            results = assess_performance(performance_df, kpis)
            avg_overall_score = results['overall_score'].mean()
            total_call_volume = results['call_volume'].sum()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Overall Score", f"{avg_overall_score:.1f}%")
            with col2:
                st.metric("Total Call Volume", f"{total_call_volume}")
            with col3:
                st.metric("Agent Count", len(results['agent_name'].unique()))
        
        tabs_list = ["📋 Set KPIs", "📝 Input Performance", "📊 Assessments", "🎯 Set Goals", "💬 Feedback", "🎙️ Audio Assessments", "🏆 Leaderboard"]
        if st.session_state.get("notifications_enabled", False):
            tabs_list.append("🌐 Community Forum")
        tabs = st.tabs(tabs_list)
        
        with tabs[0]:  # Set KPIs
            st.header("📋 Set KPI Thresholds")
            kpis = get_kpis(supabase)
            with st.form("kpi_form"):
                attendance = st.number_input("Attendance (%, min)", value=float(kpis.get('attendance', 95.0)), min_value=0.0, max_value=100.0)
                quality_score = st.number_input("Quality Score (%, min)", value=float(kpis.get('quality_score', 90.0)), min_value=0.0, max_value=100.0)
                product_knowledge = st.number_input("Product Knowledge (%, min)", value=float(kpis.get('product_knowledge', 85.0)), min_value=0.0, max_value=100.0)
                contact_success_rate = st.number_input("Contact Success Rate (%, min)", value=float(kpis.get('contact_success_rate', 80.0)), min_value=0.0, max_value=100.0)
                onboarding = st.number_input("Onboarding (%, min)", value=float(kpis.get('onboarding', 90.0)), min_value=0.0, max_value=100.0)
                reporting = st.number_input("Reporting (%, min)", value=float(kpis.get('reporting', 95.0)), min_value=0.0, max_value=100.0)
                talk_time = st.number_input("CRM Talk Time (seconds, min)", value=float(kpis.get('talk_time', 300.0)), min_value=0.0)
                resolution_rate = st.number_input("Issue Resolution Rate (%, min)", value=float(kpis.get('resolution_rate', 80.0)), min_value=0.0, max_value=100.0)
                aht = st.number_input("Average Handle Time (seconds, max)", value=float(kpis.get('aht', 600.0)), min_value=0.0)
                csat = st.number_input("Customer Satisfaction (%, min)", value=float(kpis.get('csat', 85.0)), min_value=0.0, max_value=100.0)
                call_volume = st.number_input("Call Volume (calls, min)", value=int(kpis.get('call_volume', 50)), min_value=0)
                if st.form_submit_button("Save KPIs"):
                    new_kpis = {
                        'attendance': attendance, 'quality_score': quality_score, 'product_knowledge': product_knowledge,
                        'contact_success_rate': contact_success_rate, 'onboarding': onboarding, 'reporting': reporting,
                        'talk_time': talk_time, 'resolution_rate': resolution_rate, 'aht': aht, 'csat': csat,
                        'call_volume': call_volume
                    }
                    if save_kpis(supabase, new_kpis):
                        get_kpis.clear()
                        st.success("KPIs saved!")

        with tabs[1]:  # Input Performance
            st.header("📝 Input Agent Performance")
            agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
            if not agents:
                st.warning("No agents found.")
            else:
                with st.form("performance_form"):
                    agent = st.selectbox("Select Agent", agents)
                    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
                    quality_score = st.number_input("Quality Score (%)", min_value=0.0, max_value=100.0)
                    product_knowledge = st.number_input("Product Knowledge (%)", min_value=0.0, max_value=100.0)
                    contact_success_rate = st.number_input("Contact Success Rate (%)", min_value=0.0, max_value=100.0)
                    onboarding = st.number_input("Onboarding (%)", min_value=0.0, max_value=100.0)
                    reporting = st.number_input("Reporting (%)", min_value=0.0, max_value=100.0)
                    talk_time = st.number_input("CRM Talk Time (seconds)", min_value=0.0)
                    resolution_rate = st.number_input("Issue Resolution Rate (%)", min_value=0.0, max_value=100.0)
                    aht = st.number_input("Average Handle Time (seconds)", min_value=0.0)
                    csat = st.number_input("Customer Satisfaction (%)", min_value=0.0, max_value=100.0)
                    call_volume = st.number_input("Call Volume (calls)", min_value=0)
                    if st.form_submit_button("Submit Performance"):
                        data = {
                            'attendance': attendance, 'quality_score': quality_score, 'product_knowledge': product_knowledge,
                            'contact_success_rate': contact_success_rate, 'onboarding': onboarding, 'reporting': reporting,
                            'talk_time': talk_time, 'resolution_rate': resolution_rate, 'aht': aht, 'csat': csat,
                            'call_volume': call_volume
                        }
                        if save_performance(supabase, agent, data):
                            get_performance.clear()
                            st.success(f"Performance saved for {agent}!")
            
            st.subheader("Upload Performance Data")
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file and st.button("Import CSV"):
                df = pd.read_csv(uploaded_file)
                required_cols = ['agent_name', 'attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                                'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']
                if all(col in df.columns for col in required_cols):
                    for _, row in df.iterrows():
                        data = {col: row[col] for col in required_cols[1:]}
                        if 'date' in row:
                            data['date'] = row['date']
                        save_performance(supabase, row['agent_name'], data)
                    get_performance.clear()
                    st.success(f"Imported data for {len(df)} agents!")
                else:
                    st.error("CSV missing required columns.")

        with tabs[2]:  # Assessments
            st.header("📊 Assessment Results")
            if not performance_df.empty:
                kpis = get_kpis(supabase)
                results = assess_performance(performance_df, kpis)
                st.dataframe(results)
                st.download_button(label="📥 Download Data", data=results.to_csv(index=False), file_name="performance_data.csv")
                fig = plot_performance_chart(supabase, metrics=['attendance', 'quality_score', 'csat', 'resolution_rate'])
                if fig:
                    st.plotly_chart(fig)

        with tabs[3]:  # Set Goals
            st.header("🎯 Set Agent Goals")
            agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
            if not agents:
                st.warning("No agents found.")
            else:
                with st.form("set_goals_form"):
                    agent = st.selectbox("Select Agent", agents, key="single_goal")
                    metric = st.selectbox("Metric", ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                                                    'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat',
                                                    'call_volume', 'overall_score'], key="single_metric")
                    target_value = st.number_input("Target Value", min_value=0.0, value=80.0, key="single_target")
                    if st.form_submit_button("Set Goal"):
                        if set_agent_goal(supabase, agent, metric, target_value, st.session_state.user, is_manager=True):
                            st.success(f"Goal set for {agent}!")
                
                st.subheader("Bulk Set Goals")
                with st.form("bulk_goals_form"):
                    bulk_agents = st.multiselect("Select Agents", agents, key="bulk_agents")
                    bulk_metric = st.selectbox("Metric", ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                                                        'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat',
                                                        'call_volume', 'overall_score'], key="bulk_metric")
                    bulk_target = st.number_input("Target Value", min_value=0.0, value=80.0, key="bulk_target")
                    if st.form_submit_button("Set Bulk Goals"):
                        for agent in bulk_agents:
                            set_agent_goal(supabase, agent, bulk_metric, bulk_target, st.session_state.user, is_manager=True)
                        st.success(f"Goals set for {len(bulk_agents)} agents!")
                
                st.subheader("Approve Agent-Set Goals")
                pending_goals = supabase.table("goals").select("*").eq("status", "Awaiting Approval").in_("agent_name", agents).execute()
                if pending_goals.data:
                    pending_df = pd.DataFrame(pending_goals.data)
                    for _, row in pending_df.iterrows():
                        with st.expander(f"Goal for {row['agent_name']} - {row['metric']}"):
                            st.write(f"Target Value: {row['target_value']:.1f}{' sec' if row['metric'] == 'aht' else ''}")
                            st.write(f"Created at: {row['created_at']}")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Approve", key=f"approve_{row['id']}"):
                                    if approve_goal(supabase, row['id'], st.session_state.user, approve=True):
                                        st.success(f"Goal approved for {row['agent_name']}!")
                                        st.rerun()
                            with col2:
                                if st.button("Reject", key=f"reject_{row['id']}"):
                                    if approve_goal(supabase, row['id'], st.session_state.user, approve=False):
                                        st.success(f"Goal rejected for {row['agent_name']}!")
                                        st.rerun()
                else:
                    st.info("No pending goals to approve.")
                
                st.subheader("Current Goals")
                goals_df = supabase.table("goals").select("*").in_("agent_name", agents).execute()
                if goals_df.data:
                    goals_display_df = pd.DataFrame(goals_df.data)
                    goals_display_df['target_value'] = goals_display_df.apply(
                        lambda x: f"{x['target_value']:.1f}{' sec' if x['metric'] == 'aht' else ''}", axis=1)
                    display_columns = ['agent_name', 'metric', 'target_value', 'status', 'created_at']
                    st.dataframe(goals_display_df[display_columns])
                    st.download_button(label="📥 Download Goals", data=goals_display_df.to_csv(index=False), file_name="agent_goals.csv")
                else:
                    st.info("No goals set.")

        with tabs[4]:  # Feedback
            st.header("💬 View and Respond to Agent Feedback")
            feedback_df = get_feedback(supabase)
            if not feedback_df.empty:
                feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                feedback_df['response_timestamp'] = pd.to_datetime(feedback_df['response_timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                display_columns = ['agent_name', 'message', 'created_at', 'manager_response', 'response_timestamp']
                st.subheader("Feedback History")
                st.dataframe(feedback_df[display_columns])
                st.download_button(label="📥 Download Feedback", data=feedback_df.to_csv(index=False), file_name="agent_feedback.csv")

                st.subheader("Agent Conversations")
                agents = feedback_df['agent_name'].unique()
                agents = [a for a in agents if a not in st.session_state.get('cleared_chats', set())]
                for agent in agents:
                    agent_df = feedback_df[feedback_df['agent_name'] == agent].sort_values('created_at', ascending=False)
                    with st.expander(f"{agent} ({len(agent_df)} messages)"):
                        if st.button("Clear Chat", key=f"clear_{agent}"):
                            if 'cleared_chats' not in st.session_state:
                                st.session_state.cleared_chats = set()
                            st.session_state.cleared_chats.add(agent)
                            st.rerun()
                        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                        for _, row in agent_df.iterrows():
                            st.markdown('<div class="feedback-item agent-msg">', unsafe_allow_html=True)
                            st.write(row['message'])
                            st.markdown(f'<div class="timestamp">{row["created_at"]}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            if pd.notnull(row['manager_response']):
                                st.markdown('<div class="feedback-item manager-msg">', unsafe_allow_html=True)
                                st.write(row['manager_response'])
                                st.markdown(f'<div class="timestamp">{row["response_timestamp"]}</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            if row['id'] != (agent_df.iloc[0]['id'] if not agent_df.empty else None):
                                if st.button("Reply", key=f"reply_{row['id']}"):
                                    st.session_state.reply_to_feedback_id = row['id']
                        st.markdown('</div>', unsafe_allow_html=True)

                with st.form("respond_feedback_form"):
                    if 'reply_to_feedback_id' in st.session_state:
                        feedback_id = st.session_state.reply_to_feedback_id
                        selected_feedback = feedback_df[feedback_df['id'] == feedback_id]
                        if not selected_feedback.empty:
                            st.write(f"Replying to {selected_feedback['agent_name'].iloc[0]}'s feedback: {selected_feedback['message'].iloc[0][:50]}...")
                        else:
                            st.warning("Selected feedback not found.")
                            feedback_id = None
                    else:
                        latest_feedback = feedback_df.sort_values('created_at', ascending=False).iloc[0] if not feedback_df.empty else None
                        feedback_id = latest_feedback['id'] if latest_feedback is not None and latest_feedback['agent_name'] not in st.session_state.get('cleared_chats', set()) else None
                        if feedback_id:
                            st.write(f"Replying to latest feedback from {latest_feedback['agent_name']}: {latest_feedback['message'][:50]}...")
                        else:
                            st.write("No feedback available to reply to.")
                    manager_response = st.text_area("Your Response", key="manager_response")
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        submit = st.form_submit_button("Send")
                    if submit and feedback_id and manager_response.strip():
                        if respond_to_feedback(supabase, feedback_id, manager_response, st.session_state.user):
                            get_feedback.clear()
                            st.success("Response sent!")
                            if 'reply_to_feedback_id' in st.session_state:
                                del st.session_state.reply_to_feedback_id
                            st.rerun()
                        else:
                            st.error("Failed to send response.")
                    elif submit:
                        st.error("Please provide a response and ensure a feedback is selected.")

        with tabs[5]:  # Audio Assessments
            st.header("🎙️ Audio Assessments")
            st.subheader("Upload Audio for Agent")
            agents = [user["name"] for user in supabase.table("users").select("*").eq("role", "Agent").execute().data]
            if not agents:
                st.warning("No agents found.")
            else:
                with st.form("audio_upload_form"):
                    selected_agent = st.selectbox("Select Agent", agents, key="audio_agent")
                    audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"], key="audio_file")
                    if st.form_submit_button("Upload Audio"):
                        if audio_file:
                            if upload_audio(supabase, selected_agent, audio_file, st.session_state.user):
                                get_audio_assessments.clear()
                                st.success(f"Audio uploaded for {selected_agent}!")
                            else:
                                st.error("Failed to upload audio.")
                        else:
                            st.error("Please select an audio file to upload.")

            st.subheader("Review Audio Assessments")
            audio_df = get_audio_assessments(supabase)
            if not audio_df.empty:
                audio_df['upload_timestamp'] = pd.to_datetime(audio_df['upload_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                for _, row in audio_df.iterrows():
                    with st.expander(f"{row['agent_name']} - {row['upload_timestamp']}"):
                        st.audio(row['audio_url'], format="audio/mp3")
                        st.write(f"Uploaded by: {row['uploaded_by']}")
                        notes = st.text_area("Assessment Notes", value=row['assessment_notes'], key=f"notes_{row['id']}")
                        if st.button("Save Notes", key=f"save_notes_{row['id']}"):
                            if update_assessment_notes(supabase, row['id'], notes):
                                get_audio_assessments.clear()
                                st.success("Notes saved!")
                                st.rerun()
                            else:
                                st.error("Failed to save notes.")
                st.dataframe(audio_df[['agent_name', 'upload_timestamp', 'uploaded_by', 'assessment_notes']])
                st.download_button(label="📥 Download Audio Assessments", data=audio_df.to_csv(index=False), file_name="audio_assessments.csv")
            else:
                st.info("No audio assessments available.")

        with tabs[6]:  # Leaderboard
            st.header("🏆 Leaderboard")
            leaderboard_df = get_leaderboard(supabase)
            if not leaderboard_df.empty:
                st.dataframe(leaderboard_df)
                fig = px.bar(leaderboard_df, x="agent_name", y="overall_score", color="agent_name", title="Agent Leaderboard")
                st.plotly_chart(fig)
            with st.form("award_badge_form"):
                agent = st.selectbox("Select Agent", agents)
                badge_name = st.text_input("Badge Name")
                description = st.text_area("Description")
                if st.form_submit_button("Award Badge"):
                    if award_badge(supabase, agent, badge_name, description, st.session_state.user):
                        get_leaderboard.clear()
                        st.success(f"Badge awarded to {agent}!")

        if st.session_state.get("notifications_enabled", False):
            with tabs[7]:  # Community Forum
                st.header("🌐 Community Forum")
                category = st.selectbox("Category", ["Tips", "Challenges", "General"])
                with st.form("forum_post_form"):
                    message = st.text_area("Post a Message")
                    if st.form_submit_button("Post"):
                        if create_forum_post(supabase, st.session_state.user, message, category):
                            get_forum_posts.clear()
                            st.success("Post submitted!")
                            st.rerun()
                posts_df = get_forum_posts(supabase, category)
                if not posts_df.empty:
                    for _, post in posts_df.iterrows():
                        badge_display = f" 🏅x{post['badge_count']}" if post['badge_count'] > 0 else ""
                        st.markdown(f"**{post['user_name']}{badge_display}** ({post['created_at'][:10]}): {post['message']}")
                else:
                    st.info("No posts in this category.")

    # Agent Dashboard
    elif st.session_state.role == "Agent":
        st.title(f"👤 Agent Dashboard - {st.session_state.user}")
        if st.session_state.user == "Joseph Kavuma":
            try:
                st.image("Joseph.jpg", caption="Agent Profile", width=150)
            except:
                st.error("Error loading profile image.")
        
        tabs_list = ["📋 Metrics", "🎯 Goals", "💬 Feedback", "📊 Tickets", "🏆 Achievements", "📋 Daily Report", "🤖 Ask the Coach", "🔒 Change Password"]
        if st.session_state.get("notifications_enabled", False):
            tabs_list.append("🌐 Community Forum")
        tabs = st.tabs(tabs_list)
        
        with tabs[0]:  # Metrics
            st.header("📈 Performance Metrics")
            performance_df = get_performance(supabase, st.session_state.user)
            all_performance_df = get_performance(supabase)
            if not performance_df.empty and not all_performance_df.empty:
                kpis = get_kpis(supabase)
                results = assess_performance(performance_df, kpis)
                all_results = assess_performance(all_performance_df, kpis)
                avg_overall_score = results['overall_score'].mean()
                avg_metrics = results[['overall_score', 'quality_score', 'csat', 'attendance', 
                                     'resolution_rate', 'contact_success_rate', 'aht', 'talk_time']].mean()
                total_call_volume = results['call_volume'].sum()
                
                if avg_overall_score > 90:
                    st.markdown('<span style="color: gold; font-weight: bold;">🏆 Top Performer</span>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{avg_metrics['overall_score']:.1f}%")
                    st.metric("Quality Score", f"{avg_metrics['quality_score']:.1f}%")
                    st.metric("Customer Satisfaction", f"{avg_metrics['csat']:.1f}%")
                with col2:
                    st.metric("Attendance", f"{avg_metrics['attendance']:.1f}%")
                    st.metric("Resolution Rate", f"{avg_metrics['resolution_rate']:.1f}%")
                    st.metric("Contact Success", f"{avg_metrics['contact_success_rate']:.1f}%")
                with col3:
                    st.metric("Average Handle Time", f"{avg_metrics['aht']:.1f} sec")
                    st.metric("Talk Time", f"{avg_metrics['talk_time']:.1f} sec")
                    st.metric("Call Volume", f"{total_call_volume:.0f} calls")
                
                st.subheader("Performance Profile")
                fig = plot_performance_chart(supabase, st.session_state.user)
                if fig:
                    st.plotly_chart(fig)
                
                st.subheader("Comparison to Peers")
                peer_avg = all_results.groupby('agent_name')['overall_score'].mean().reset_index()
                peer_avg = peer_avg[peer_avg['agent_name'] != st.session_state.user]
                fig3 = px.box(peer_avg, y='overall_score', title="Peer Score Distribution", labels={'overall_score': 'Score (%)'}, points="all")
                fig3.add_hline(y=avg_overall_score, line_dash="dash", line_color="red", annotation_text=f"Your Score: {avg_overall_score:.1f}%")
                st.plotly_chart(fig3)
                
                st.subheader("🤖 Coaching Tips")
                tips = get_coaching_tips(supabase, st.session_state.user)
                if tips:
                    for tip in tips:
                        st.markdown(f"**{tip['metric'].replace('_', ' ').title()}**: {tip['tip']}")
                else:
                    st.info("You're performing well! Keep it up.")

        with tabs[1]:  # Goals
            st.header("🎯 Your Goals")
            all_metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                          'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat',
                          'call_volume', 'overall_score']
            response = supabase.table("goals").select("*").eq("agent_name", st.session_state.user).execute()
            goals_df = pd.DataFrame(response.data)
            if not goals_df.empty:
                for metric in all_metrics:
                    goal_row = goals_df[goals_df['metric'] == metric]
                    if not goal_row.empty:
                        row = goal_row.iloc[0]
                        current_value = results[results['date'] == max(results['date'])][metric].mean() if metric in results.columns else 0.0
                        progress = min((kpis.get(metric, 600) - current_value) / (kpis.get(metric, 600) - row['target_value']) * 100, 100) if metric == 'aht' else min(current_value / row['target_value'] * 100, 100) if row['target_value'] > 0 else 0
                        color = "green" if progress >= 80 else "orange" if progress >= 50 else "red"
                        st.markdown(f"<div class='progress-bar' style='background-color: {color}; width: {progress}%;'></div>", unsafe_allow_html=True)
                        st.write(f"{metric.replace('_', ' ').title()}: Target {row['target_value']:.1f}{' sec' if metric == 'aht' else '%'}, Current {current_value:.1f}{' sec' if metric == 'aht' else '%'}, Status: {row['status']}")
                        if row['status'] in ["Pending", "Awaiting Approval"]:
                            with st.form(f"update_goal_form_{metric}"):
                                new_target = st.number_input(f"New Target for {metric}", value=float(row['target_value']), key=f"new_target_{metric}")
                                if st.form_submit_button(f"Update {metric} Goal"):
                                    if set_agent_goal(supabase, st.session_state.user, metric, new_target, st.session_state.user, is_manager=False):
                                        st.success(f"Goal update submitted for {metric}! Awaiting manager approval.")
                    else:
                        with st.form(f"set_goal_form_{metric}"):
                            target_value = st.number_input(f"Set Target for {metric}", min_value=0.0, value=80.0, key=f"set_target_{metric}")
                            if st.form_submit_button(f"Set {metric} Goal"):
                                if set_agent_goal(supabase, st.session_state.user, metric, target_value, st.session_state.user, is_manager=False):
                                    st.success(f"Goal submitted for {metric}! Awaiting manager approval.")
            else:
                for metric in all_metrics:
                    with st.form(f"set_goal_form_{metric}"):
                        target_value = st.number_input(f"Set Target for {metric}", min_value=0.0, value=80.0, key=f"set_target_{metric}")
                        if st.form_submit_button(f"Set {metric} Goal"):
                            if set_agent_goal(supabase, st.session_state.user, metric, target_value, st.session_state.user, is_manager=False):
                                st.success(f"Goal submitted for {metric}! Awaiting manager approval.")

        with tabs[2]:  # Feedback
            st.header("💬 Feedback and Responses")
            with st.form("feedback_form"):
                feedback_text = st.text_area("Submit Feedback")
                if st.form_submit_button("Submit Feedback"):
                    supabase.table("feedback").insert({
                        "agent_name": st.session_state.user,
                        "message": feedback_text
                    }).execute()
                    if st.session_state.get("notifications_enabled", False):
                        managers = supabase.table("users").select("id").eq("role", "Manager").execute()
                        for manager in managers.data:
                            supabase.table("notifications").insert({
                                "user_id": manager["id"],
                                "message": f"New feedback from {st.session_state.user}: {feedback_text[:50]}..."
                            }).execute()
                    get_feedback.clear()
                    st.success("Feedback submitted!")
            
            st.subheader("Feedback History")
            feedback_df = get_feedback(supabase, st.session_state.user)
            if not feedback_df.empty:
                feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                feedback_df['response_timestamp'] = pd.to_datetime(feedback_df['response_timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                display_columns = ['message', 'created_at', 'manager_response', 'response_timestamp']
                st.dataframe(feedback_df[display_columns])
                st.download_button(label="📥 Download Feedback", data=feedback_df.to_csv(index=False), file_name="feedback_history.csv")
            else:
                st.info("No feedback submitted.")

        with tabs[3]:  # Tickets
            st.header("📊 Zoho Ticket Data")
            zoho_df = get_zoho_agent_data(supabase, st.session_state.user)
            if not zoho_df.empty:
                total_tickets = zoho_df['id'].nunique()
                st.metric("Total Tickets Handled", f"{total_tickets}")
                time_col = None
                if 'created_time' in zoho_df.columns:
                    time_col = 'created_time'
                elif 'created_at' in zoho_df.columns:
                    time_col = 'created_at'
                else:
                    st.warning("No 'created_time' or 'created_at' column found in Zoho data.")
                
                if time_col:
                    try:
                        zoho_df[time_col] = pd.to_datetime(zoho_df[time_col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        st.warning("Error formatting time column.")
                
                display_cols = ['id', 'subject', 'status']
                if time_col:
                    display_cols.append(time_col)
                display_cols.append('priority')
                display_cols = [col for col in display_cols if col in zoho_df.columns]
                
                st.dataframe(zoho_df[display_cols])
                channel_counts = zoho_df.groupby('channel')['id'].nunique().reset_index(name='Ticket Count')
                st.subheader("Ticket Breakdown by Channel")
                st.dataframe(channel_counts)
                try:
                    fig = px.pie(channel_counts, values='Ticket Count', names='channel', title="Ticket Distribution by Channel")
                    st.plotly_chart(fig)
                except Exception:
                    st.error("Error plotting ticket distribution.")
                st.download_button(label="📥 Download Zoho Data", data=zoho_df.to_csv(index=False), file_name="zoho_agent_data.csv")
            else:
                st.info("No Zoho data available.")

        with tabs[4]:  # Achievements
            st.header("🏆 My Achievements")
            badges_df = supabase.table("badges").select("*").eq("agent_name", st.session_state.user).execute()
            completed_goals = supabase.table("goals").select("*").eq("agent_name", st.session_state.user).eq("status", "Completed").execute()
            if badges_df.data or completed_goals.data:
                if badges_df.data:
                    for badge in badges_df.data:
                        st.markdown(f"🎖️ **{badge['badge_name']}**: {badge['description']} (Earned on {badge['earned_at'][:10]})")
                if completed_goals.data:
                    st.subheader("Completed Goals")
                    for goal in completed_goals.data:
                        st.markdown(f"🎯 **{goal['metric'].replace('_', ' ').title()} Goal Achieved**: Target {goal['target_value']:.1f}{' sec' if goal['metric'] == 'aht' else '%'} (Completed on {goal['approved_at'][:10] if goal['approved_at'] else goal['created_at'][:10]})")
            else:
                st.info("No badges or completed goals yet. Keep up the great work!")

        with tabs[5]:  # Daily Report
            st.header("📋 Call Centre Daily Report")
            st.markdown("""
            Welcome to the **Call Centre Daily Reporting Tool**.  
            Please take a moment to complete your daily update. Your input drives our growth! 🚀
            """)
            with st.spinner('Loading the reporting form...'):
                time.sleep(2)  # Simulate loading time
            # Pre-fill the "Your Name" field with the logged-in agent's name
            base_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSfWt6PzEoYv2lSL8H6WGZaL0IsDmq3I79aMWt5VOseL6CN7_Q/viewform"
            encoded_name = urllib.parse.quote(st.session_state.user)
            form_url = f"{base_form_url}?entry.1234567890={encoded_name}"
            st.markdown(
                f"""
                <iframe src="{form_url}" width="720" height="1600" frameborder="0" marginheight="0" marginwidth="0" style="background: white;">
                    Loading…
                </iframe>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")
            st.success("✅ After submitting your form, thank you for your dedication today!")
            st.caption("© 2025 BodaBoda Union | Powered by Love and Togetherness 💚")

        with tabs[6]:  # Ask the Coach
            st.header("🤖 Ask the Coach")
            with st.form("ask_coach_form"):
                question = st.text_area("Ask a question about improving your performance (e.g., 'How can I improve my CSAT?')")
                if st.form_submit_button("Ask"):
                    if question.strip():
                        answer = ask_coach(supabase, st.session_state.user, question)
                        st.markdown(f"**Coach Response**: {answer}")
                        st.session_state.last_coach_answer = answer  # Store for persistence
                    else:
                        st.error("Please enter a question.")

        if st.session_state.get("notifications_enabled", False):
            with tabs[7]:  # Community Forum
                st.header("🌐 Community Forum")
                category = st.selectbox("Category", ["Tips", "Challenges", "General"], key="agent_forum_category")
                with st.form("agent_forum_post_form"):
                    message = st.text_area("Post a Message", key="agent_forum_message")
                    if st.form_submit_button("Post"):
                        if create_forum_post(supabase, st.session_state.user, message, category):
                            get_forum_posts.clear()
                            st.success("Post submitted!")
                            st.rerun()
                posts_df = get_forum_posts(supabase, category)
                if not posts_df.empty:
                    for _, post in posts_df.iterrows():
                        badge_display = f" 🏅x{post['badge_count']}" if post['badge_count'] > 0 else ""
                        st.markdown(f"**{post['user_name']}{badge_display}** ({post['created_at'][:10]}): {post['message']}")
                else:
                    st.info("No posts in this category.")

if __name__ == "__main__":
    main()
