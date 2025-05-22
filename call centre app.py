import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import urllib.parse
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Initialization
def init_supabase() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Supabase initialization failed: {str(e)}")
        raise

def check_db(supabase: Client) -> bool:
    try:
        tables = ["users", "performance", "kpis", "goals", "feedback", "badges", "notifications"]
        for table in tables:
            response = supabase.table(table).select("*").limit(1).execute()
            if not response.data:
                logger.warning(f"Table {table} is empty or inaccessible.")
        return True
    except Exception as e:
        logger.error(f"Database check failed: {str(e)}")
        return False

# Authentication
def authenticate_user(supabase: Client, name: str, password: str) -> tuple[bool, str, str]:
    try:
        response = supabase.table("users").select("name, role, password").eq("name", name).eq("password", password).execute()
        if response.data:
            return True, response.data[0]["name"], response.data[0]["role"]
        return False, "", ""
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return False, "", ""

def change_password(supabase: Client, name: str, new_password: str) -> bool:
    try:
        supabase.table("users").update({"password": new_password}).eq("name", name).execute()
        return True
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        return False

# Data Functions
@st.cache_data(ttl=300)
def get_performance(_supabase: Client, agent_name: str = None) -> pd.DataFrame:
    try:
        query = _supabase.table("performance").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            expected_cols = [
                'agent_name', 'attendance', 'quality_score', 'product_knowledge',
                'contact_success_rate', 'onboarding', 'reporting', 'talk_time',
                'resolution_rate', 'aht', 'csat', 'call_volume', 'date'
            ]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns in performance data: {missing_cols}")
                return pd.DataFrame()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
                if df['date'].isna().any():
                    st.warning(f"Invalid 'date' values found in performance data for {agent_name or 'all agents'}.")
                    df = df[df['date'].notna()].copy()
            st.write(f"DEBUG: get_performance for {agent_name or 'all agents'}, shape: {df.shape}, date dtype: {df['date'].dtype}")
            return df
        st.warning(f"No performance data found for {agent_name or 'all agents'}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving performance data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_feedback(_supabase: Client, agent_name: str = None) -> pd.DataFrame:
    try:
        query = _supabase.table("feedback").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
            df['response_timestamp'] = pd.to_datetime(df['response_timestamp'], errors='coerce', utc=True)
            invalid_rows = df[df['created_at'].isna()]
            if not invalid_rows.empty:
                st.warning(f"Invalid 'created_at' values found for {agent_name or 'all agents'}: {invalid_rows.index.tolist()}")
            df = df[df['created_at'].notna()].copy()
            if df.empty:
                st.warning(f"No valid feedback found for {agent_name or 'all agents'} after date validation.")
            else:
                st.write(f"DEBUG: get_feedback for {agent_name or 'all agents'}, created_at dtype: {df['created_at'].dtype}")
                st.write(f"DEBUG: get_feedback created_at sample: {df['created_at'].head().tolist()}")
            return df
        st.warning(f"No feedback found for {agent_name or 'all agents'}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving feedback: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_kpis(_supabase: Client) -> dict:
    try:
        response = _supabase.table("kpis").select("*").execute()
        if response.data:
            return response.data[0]
        return {
            'attendance': 95.0, 'quality_score': 90.0, 'product_knowledge': 85.0,
            'contact_success_rate': 80.0, 'onboarding': 90.0, 'reporting': 95.0,
            'talk_time': 300.0, 'resolution_rate': 80.0, 'aht': 600.0,
            'csat': 85.0, 'call_volume': 50
        }
    except Exception as e:
        st.error(f"Error retrieving KPIs: {str(e)}")
        return {}

def save_kpis(supabase: Client, kpis: dict) -> bool:
    try:
        supabase.table("kpis").upsert(kpis).execute()
        return True
    except Exception as e:
        st.error(f"Error saving KPIs: {str(e)}")
        return False

def save_performance(supabase: Client, agent: str, data: dict) -> bool:
    try:
        data['agent_name'] = agent
        data['date'] = datetime.now(timezone.utc).isoformat()
        supabase.table("performance").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving performance: {str(e)}")
        return False

def set_agent_goal(supabase: Client, agent: str, metric: str, target: float, set_by: str, is_manager: bool) -> bool:
    try:
        status = "Approved" if is_manager else "Awaiting Approval"
        supabase.table("goals").insert({
            "agent_name": agent,
            "metric": metric,
            "target_value": target,
            "status": status,
            "set_by": set_by,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error setting goal: {str(e)}")
        return False

def approve_goal(supabase: Client, goal_id: str, approved_by: str, approve: bool) -> bool:
    try:
        status = "Approved" if approve else "Rejected"
        supabase.table("goals").update({
            "status": status,
            "approved_by": approved_by,
            "approved_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", goal_id).execute()
        return True
    except Exception as e:
        st.error(f"Error approving goal: {str(e)}")
        return False

def respond_to_feedback(supabase: Client, feedback_id: str, response: str, manager: str) -> bool:
    try:
        supabase.table("feedback").update({
            "manager_response": response,
            "response_timestamp": datetime.now(timezone.utc).isoformat()
        }).eq("id", feedback_id).execute()
        return True
    except Exception as e:
        st.error(f"Error responding to feedback: {str(e)}")
        return False

@st.cache_data(ttl=300)
def get_notifications(_supabase: Client) -> pd.DataFrame:
    try:
        response = _supabase.table("notifications").select("*").eq("read", False).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error retrieving notifications: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_audio_assessments(_supabase: Client) -> pd.DataFrame:
    try:
        response = _supabase.table("audio_assessments").select("*").execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error retrieving audio assessments: {str(e)}")
        return pd.DataFrame()

def upload_audio(supabase: Client, agent: str, audio_file, uploaded_by: str) -> bool:
    try:
        audio_url = "placeholder_url"  # Implement actual storage logic
        supabase.table("audio_assessments").insert({
            "agent_name": agent,
            "audio_url": audio_url,
            "uploaded_by": uploaded_by,
            "upload_timestamp": datetime.now(timezone.utc).isoformat()
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error uploading audio: {str(e)}")
        return False

def update_assessment_notes(supabase: Client, assessment_id: str, notes: str) -> bool:
    try:
        supabase.table("audio_assessments").update({"assessment_notes": notes}).eq("id", assessment_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating assessment notes: {str(e)}")
        return False

@st.cache_data(ttl=300)
def get_leaderboard(_supabase: Client) -> pd.DataFrame:
    try:
        performance_df = get_performance(_supabase)
        if not performance_df.empty:
            return performance_df.groupby("agent_name")[['overall_score', 'call_volume']].mean().reset_index()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving leaderboard: {str(e)}")
        return pd.DataFrame()

def award_badge(supabase: Client, agent: str, badge_name: str, description: str, awarded_by: str) -> bool:
    try:
        supabase.table("badges").insert({
            "agent_name": agent,
            "badge_name": badge_name,
            "description": description,
            "awarded_by": awarded_by,
            "earned_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error awarding badge: {str(e)}")
        return False

@st.cache_data(ttl=300)
def get_forum_posts(_supabase: Client, category: str) -> pd.DataFrame:
    try:
        response = _supabase.table("forum_posts").select("*").eq("category", category).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error retrieving forum posts: {str(e)}")
        return pd.DataFrame()

def create_forum_post(supabase: Client, user: str, message: str, category: str) -> bool:
    try:
        supabase.table("forum_posts").insert({
            "user_name": user,
            "message": message,
            "category": category,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error creating forum post: {str(e)}")
        return False

@st.cache_data(ttl=300)
def get_zoho_agent_data(_supabase: Client, agent: str) -> pd.DataFrame:
    try:
        response = _supabase.table("zoho_tickets").select("*").eq("agent_name", agent).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error retrieving Zoho data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_coaching_tips(_supabase: Client, agent: str) -> list:
    try:
        performance_df = get_performance(_supabase, agent)
        if performance_df.empty:
            return []
        kpis = get_kpis(_supabase)
        tips = []
        for metric, value in performance_df.mean().items():
            if metric in kpis and value < kpis[metric]:
                tips.append({"metric": metric, "tip": f"Improve {metric.replace('_', ' ')} by focusing on consistency."})
        return tips
    except Exception as e:
        st.error(f"Error retrieving coaching tips: {str(e)}")
        return []

def calculate_goal_progress(goal: dict, performance_df: pd.DataFrame) -> float:
    try:
        metric = goal['metric']
        target = goal['target_value']
        if not performance_df.empty and metric in performance_df.columns:
            current = performance_df[metric].mean()
            return min((current / target) * 100, 100.0)
        return 0.0
    except Exception as e:
        st.error(f"Error calculating goal progress: {str(e)}")
        return 0.0

def assess_performance(performance_df: pd.DataFrame, kpis: dict) -> pd.DataFrame:
    try:
        results = performance_df.copy()
        weights = {
            'attendance': 0.15, 'quality_score': 0.2, 'product_knowledge': 0.1,
            'contact_success_rate': 0.1, 'onboarding': 0.05, 'reporting': 0.05,
            'talk_time': 0.1, 'resolution_rate': 0.15, 'csat': 0.1
        }
        results['overall_score'] = sum(
            results[metric].clip(0, 100) * weight for metric, weight in weights.items()
        )
        return results
    except Exception as e:
        st.error(f"Error assessing performance: {str(e)}")
        return pd.DataFrame()

def plot_performance_chart(supabase: Client, agent: str = None, metrics: list = None) -> px.Figure:
    try:
        df = get_performance(supabase, agent)
        if df.empty:
            return None
        if metrics:
            df = df[metrics + ['agent_name']]
        fig = px.line(df, x='date', y=metrics or df.columns.drop(['agent_name', 'date']), color='agent_name')
        return fig
    except Exception as e:
        st.error(f"Error plotting performance chart: {str(e)}")
        return None

def setup_realtime(supabase: Client):
    try:
        def handle_update(payload):
            st.session_state.data_updated = True
        supabase.realtime.connect()
        supabase.table("performance").on("INSERT", handle_update).subscribe()
        supabase.table("feedback").on("INSERT", handle_update).subscribe()
    except Exception as e:
        st.error(f"Error setting up realtime: {str(e)}")

def generate_pdf_report(supabase: Client, agents: list, start_date, end_date, metrics: list) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle(name='Title', fontSize=16, leading=20, alignment=1, spaceAfter=20)
    subtitle_style = ParagraphStyle(name='Subtitle', fontSize=12, leading=15, spaceAfter=10)
    normal_style = styles['Normal']

    elements.append(Paragraph("Call Center Agent Performance Report", title_style))
    elements.append(Paragraph(f"Generated for agents: {', '.join(agents)}", normal_style))
    elements.append(Spacer(1, 12))

    for agent in agents:
        elements.append(Paragraph(f"Agent: {agent}", subtitle_style))
        elements.append(Spacer(1, 12))

        perf_df = get_performance(supabase, agent)
        if not perf_df.empty:
            perf_data = perf_df[metrics].mean().to_dict()
            table_data = [['Metric', 'Value']] + [
                [metric.replace('_', ' ').title(), f"{value:.1f}{' sec' if metric == 'aht' else '%' if metric != 'call_volume' else ''}"]
                for metric, value in perf_data.items()
            ]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(Paragraph("Performance Metrics", normal_style))
            elements.append(table)
            elements.append(Spacer(1, 12))

        goals_df = pd.DataFrame(supabase.table("goals").select("*").eq("agent_name", agent).execute().data)
        if not goals_df.empty:
            elements.append(Paragraph("Goals", normal_style))
            for _, goal in goals_df.iterrows():
                status = goal['status']
                target = f"{goal['target_value']:.1f}{' sec' if goal['metric'] == 'aht' else ''}"
                elements.append(Paragraph(f"{goal['metric'].replace('_', ' ').title()}: Target {target}, Status: {status}", normal_style))
            elements.append(Spacer(1, 12))

        badges_df = pd.DataFrame(supabase.table("badges").select("*").eq("agent_name", agent).execute().data)
        if not badges_df.empty:
            elements.append(Paragraph("Badges", normal_style))
            for _, badge in badges_df.iterrows():
                elements.append(Paragraph(f"{badge['badge_name']}: {badge['description']} (Earned on {badge['earned_at'][:10]})", normal_style))
            elements.append(Spacer(1, 12))

        feedback_df = get_feedback(supabase, agent)
        if not feedback_df.empty:
            elements.append(Paragraph("Feedback", normal_style))
            for _, feedback in feedback_df.iterrows():
                elements.append(Paragraph(f"Feedback: {feedback['message']} (Submitted on {feedback['created_at'].strftime('%Y-%m-%d')})", normal_style))
                if pd.notnull(feedback['manager_response']):
                    response_timestamp = pd.to_datetime(feedback['response_timestamp'], errors='coerce', utc=True)
                    response_date = response_timestamp.strftime('%Y-%m-%d') if pd.notnull(response_timestamp) else "Unknown"
                    elements.append(Paragraph(f"Response: {feedback['manager_response']} (Responded on {response_date})", normal_style))
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("Feedback: No feedback available.", normal_style))
            elements.append(Spacer(1, 12))

        elements.append(Spacer(1, 20))

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawString(50, 30, f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | BodaBoda Union")
        canvas.restoreState()

    try:
        doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
        buffer.seek(0)
        if buffer.getvalue() == b"":
            st.error("PDF generation failed: Buffer is empty.")
            return None
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        logger.error(f"PDF generation error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    
    try:
        supabase = init_supabase()
        if not check_db(supabase):
            st.error("Critical database tables are missing.")
            st.stop()
        global auth
        auth = supabase.auth
        st.session_state.supabase = supabase
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        st.stop()

    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.data_updated = False
        st.session_state.notifications_enabled = False
        st.session_state.auto_refresh = False
        st.session_state.last_refresh = datetime.now()
        st.session_state.cleared_chats = set()
        st.session_state.theme = "light"
        st.session_state.authenticated = False

    theme_css = {
        "light": """
            .reportview-container { background: linear-gradient(to right, #f0f4f8, #e0e7ff); }
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
            .reportview-container { background: linear-gradient(to right, #2c3e50, #34495e); color: white; }
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
            .reportview-container { background: linear-gradient(to right, #3498db, #2980b9); color: white; }
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
    st.markdown(f"<style>{theme_css[st.session_state.theme]}</style>", unsafe_allow_html=True)

    with st.sidebar:
        theme = st.selectbox("Theme", ["Light", "Dark", "Blue"], key="theme_selector")
        if theme.lower() != st.session_state.theme:
            st.session_state.theme = theme.lower()
            st.rerun()
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    if not st.session_state.user:
        st.title("🔐 Login")
        with st.form("login_form"):
            name = st.text_input("Name")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                success, user, role = authenticate_user(supabase, name, password)
                if success:
                    st.session_state.user = user
                    st.session_state.role = role
                    st.session_state.authenticated = True
                    st.success(f"Logged in as {user} ({role})")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        return

    with st.sidebar:
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.role = None
            st.session_state.authenticated = False
            st.session_state.clear()
            st.write("DEBUG: Session cleared on logout.")
            st.rerun()

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

    st.session_state.auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=st.session_state.get("auto_refresh", False))
    setup_realtime(supabase)
    if st.session_state.get("auto_refresh", False) and st.session_state.get("data_updated", False):
        st.session_state.data_updated = False
        st.rerun()

    st.sidebar.info(f"👤 Logged in as: {st.session_state.user}")
    st.sidebar.info(f"🎓 Role: {st.session_state.role}")

    try:
        st.image(r"./companylogo.png", width=150)
    except Exception:
        st.warning("Failed to load company logo.")

    if st.session_state.role == "Manager":
        st.title("📊 Manager Dashboard")
        performance_df = get_performance(supabase)
        if performance_df.empty:
            st.warning("No performance data available. Please input performance data or check the database.")
        else:
            try:
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
            except Exception as e:
                st.error(f"Error processing performance data: {str(e)}")

        tabs_list = ["📋 Set KPIs", "📝 Input Performance", "📊 Assessments", "🎯 Set Goals", "💬 Feedback", "🎙️ Audio Assessments", "🏆 Leaderboard"]
        if st.session_state.get("notifications_enabled", False):
            tabs_list.append("🌐 Community Forum")
        tabs = st.tabs(tabs_list)

        with tabs[0]:
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

        with tabs[1]:
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

        with tabs[2]:
            st.header("📊 Assessment Results")
            if not performance_df.empty:
                kpis = get_kpis(supabase)
                results = assess_performance(performance_df, kpis)
                st.dataframe(results)
                st.download_button(label="📥 Download Data", data=results.to_csv(index=False), file_name="performance_data.csv")
                fig = plot_performance_chart(supabase, metrics=['attendance', 'quality_score', 'csat', 'resolution_rate'])
                if fig:
                    st.plotly_chart(fig)

                st.subheader("Generate Custom Report")
                with st.form("custom_report_form"):
                    agents = st.multiselect("Select Agents", results['agent_name'].unique(), default=results['agent_name'].unique())
                    start_date = st.date_input("Start Date (Optional)", value=datetime.now() - timedelta(days=30))
                    end_date = st.date_input("End Date (Optional)", value=datetime.now())
                    available_metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                                        'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']
                    selected_metrics = st.multiselect("Select Metrics", available_metrics, default=['attendance', 'quality_score', 'csat', 'aht'])
                    if st.form_submit_button("Generate PDF Report"):
                        if agents and selected_metrics:
                            try:
                                pdf_buffer = generate_pdf_report(supabase, agents, start_date, end_date, selected_metrics)
                                if pdf_buffer is not None and pdf_buffer.getvalue():
                                    st.download_button(
                                        label="📥 Download PDF Report",
                                        data=pdf_buffer,
                                        file_name=f"agent_performance_report.pdf",
                                        mime="application/pdf"
                                    )
                                    st.success("PDF report generated successfully!")
                                else:
                                    st.error("Failed to generate PDF report. Please check the data and try again.")
                            except Exception as e:
                                st.error(f"Error generating PDF report: {str(e)}")
                        else:
                            st.error("Please select at least one agent and one metric.")
            else:
                st.info("No performance data available to display.")

        with tabs[3]:
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

        with tabs[4]:
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

        with tabs[5]:
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

        with tabs[6]:
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
            with tabs[7]:
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

    elif st.session_state.role == "Agent":
        st.title(f"👤 Agent Dashboard - {st.session_state.user}")
        st.write(f"DEBUG: Notifications Enabled: {st.session_state.get('notifications_enabled', False)}")
        if st.session_state.user == "Joseph Kavuma":
            try:
                st.image("Joseph.jpg", caption="Agent Profile", width=150)
            except:
                st.error("Error loading profile image.")

        tabs_list = ["📋 Metrics", "🎯 Goals", "💬 Feedback", "📊 Tickets", "🏆 Achievements", "📋 Daily Report", "🤖 Ask the Coach", "🔒 Change Password"]
        if st.session_state.get("notifications_enabled", False):
            tabs_list.append("🌐 Community Forum")
        tabs = st.tabs(tabs_list)

        with tabs[0]:
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
                    st.info("You're performing well! No coaching tips needed.")
            else:
                st.info("No performance data available.")

        with tabs[1]:
            st.header("🎯 Your Goals")
            all_metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                          'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat',
                          'call_volume', 'overall_score']
            response = supabase.table("goals").select("*").eq("agent_name", st.session_state.user).execute()
            goals_df = pd.DataFrame(response.data)
            if not goals_df.empty:
                goals_df['progress'] = goals_df.apply(
                    lambda row: calculate_goal_progress(row, get_performance(supabase, st.session_state.user)), axis=1)
                for _, goal in goals_df.iterrows():
                    with st.container():
                        st.write(f"**{goal['metric'].replace('_', ' ').title()}**: Target {goal['target_value']:.1f}{' sec' if goal['metric'] == 'aht' else ''}")
                        st.write(f"Status: {goal['status']}")
                        progress = min(goal['progress'], 100.0)
                        color = "green" if progress >= 80 else "orange" if progress >= 50 else "red"
                        st.markdown(f"<div class='progress-bar' style='background-color: {color}; width: {progress}%;'></div>", unsafe_allow_html=True)
                        if goal['status'] in ["Pending", "Awaiting Approval"]:
                            with st.form(f"update_goal_form_{goal['metric']}"):
                                new_target = st.number_input(f"New Target for {goal['metric']}", value=float(goal['target_value']), key=f"new_target_{goal['metric']}")
                                if st.form_submit_button(f"Update {goal['metric']} Goal"):
                                    if set_agent_goal(supabase, st.session_state.user, goal['metric'], new_target, st.session_state.user, is_manager=False):
                                        st.success(f"Goal update submitted for {goal['metric']}!")
                                        st.rerun()
            for metric in all_metrics:
                if metric not in goals_df['metric'].values:
                    with st.form(f"set_goal_form_{metric}"):
                        target_value = st.number_input(f"Set Target for {metric}", min_value=0.0, value=80.0, key=f"set_target_{metric}")
                        if st.form_submit_button(f"Set {metric} Goal"):
                            if set_agent_goal(supabase, st.session_state.user, metric, target_value, st.session_state.user, is_manager=False):
                                st.success(f"Goal submitted for {metric}! Awaiting manager approval.")
                                st.rerun()

        with tabs[2]:
            st.header("💬 Feedback and Responses")
            with st.form("feedback_form"):
                feedback_text = st.text_area("Submit Feedback")
                if st.form_submit_button("Submit Feedback"):
                    if feedback_text.strip():
                        supabase.table("feedback").insert({
                            "agent_name": st.session_state.user,
                            "message": feedback_text,
                            "created_at": datetime.now(timezone.utc).isoformat()
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
                        st.rerun()
                    else:
                        st.error("Please enter feedback text.")

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

        with tabs[3]:
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

        with tabs[4]:
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

        with tabs[5]:
            st.header("📋 Call Centre Daily Report")
            st.markdown("""
            Welcome to the **Call Centre Daily Reporting Tool**.
            Please take a moment to complete your daily update. Your input drives our growth! 🚀
            """)
            with st.spinner('Loading the reporting form...'):
                time.sleep(2)
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

        with tabs[6]:
            st.header("🤖 Ask the Coach")
            with st.form("ask_coach_form"):
                question = st.text_area("Ask a question about improving your performance (e.g., 'How can I improve my CSAT?')")
                if st.form_submit_button("Ask"):
                    if question.strip():
                        answer = get_coaching_tips(supabase, st.session_state.user)  # Simplified for example
                        st.markdown(f"**Coach Response**: {answer}")
                        st.session_state.last_coach_answer = answer
                    else:
                        st.error("Please enter a question.")

        with tabs[7]:
            st.header("🔒 Change Password")
            with st.form("change_password_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")
                if st.form_submit_button("Change Password"):
                    if not new_password or not confirm_password:
                        st.error("Please enter both new password and confirmation.")
                    elif new_password != confirm_password:
                        st.error("New password and confirmation do not match.")
                    else:
                        success, _, _ = authenticate_user(supabase, st.session_state.user, current_password)
                        if not success:
                            st.error("Current password is incorrect.")
                        elif change_password(supabase, st.session_state.user, new_password):
                            st.success("Password changed successfully! Please log in again.")
                            st.session_state.user = None
                            st.session_state.role = None
                            st.session_state.authenticated = False
                            st.session_state.clear()
                            st.write("DEBUG: Session cleared after password change.")
                            st.rerun()
                        else:
                            st.error("Failed to change password.")

        if st.session_state.get("notifications_enabled", False):
            with tabs[8]:
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
