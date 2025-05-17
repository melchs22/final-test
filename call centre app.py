
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client
import uuid
import re
import io

# Agent extension mapping
AGENT_EXTENSIONS = {
    "1004": "Melchizedek Tutu",
    "1003": "Joseph Kavuma",
    "1001": "Daisy Nahabwe",
    "1005": "Cynthia Lunkuse",
    "1006": "Amulet Kyokusiima",
    "1010": "Oyo Jacob Humphrey"
}

def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        if not url.startswith("https://"):
            url = f"https://{url}"
        client = create_client(url, key)
        # Check if using anonymous key
        if key == st.secrets["supabase"].get("anon_key", ""):
            st.warning("Using anonymous Supabase key. RLS policies for 'authenticated' role may fail.")
        return client
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        raise e

def authenticate_user(supabase, name, password):
    # TODO: Replace with Supabase Authentication for secure password handling
    # See https://supabase.com/docs/guides/auth
    try:
        response = supabase.table("users").select("id, name, role, password").eq("name", name).execute()
        if not response.data:
            return False, None, None, None
        user_data = response.data[0]
        if user_data["password"] == password:
            # Simulate setting session for authenticated user (replace with auth.sign_in_with_password)
            st.session_state.supabase_user_id = user_data["id"]
            return True, user_data["name"], user_data["role"], user_data["id"]
        else:
            return False, None, None, None
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return False, None, None, None

def set_supabase_session(supabase, user_id):
    # Placeholder for setting authenticated session
    # In production, use supabase.auth.sign_in_with_password to get JWT
    # For now, store user_id in session state for RLS debugging
    try:
        # Debug: Check current user context
        st.session_state.supabase_user_id = user_id
        # TODO: Implement JWT-based authentication
        # response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        # supabase.auth.set_session(response.session.access_token)
    except Exception as e:
        st.error(f"Failed to set Supabase session: {str(e)}")

def check_db(supabase):
    required_tables = ["users", "kpis", "performance", "zoho_agent_data", "goals", "feedback", "notifications", "audio_assessments", "cdr_reports"]
    critical_tables = ["users", "goals", "feedback", "performance"]
    missing_critical = []
    missing_non_critical = []
    
    for table in required_tables:
        try:
            supabase.table(table).select("count").limit(1).execute()
        except Exception as e:
            if 'relation' in str(e).lower() and 'does not exist' in str(e).lower():
                if table in critical_tables:
                    missing_critical.append(table)
                else:
                    missing_non_critical.append(table)
            else:
                st.sidebar.warning(f"Error accessing {table}: {str(e)}")
    
    if missing_critical:
        st.sidebar.error(f"Critical tables missing: {', '.join(missing_critical)}. Please create them to use the app.")
        return False
    if missing_non_critical:
        st.sidebar.warning(f"Non-critical tables missing: {', '.join(missing_non_critical)}. Some features may be unavailable.")
        if "notifications" in missing_non_critical:
            st.session_state.notifications_enabled = False
        else:
            st.session_state.notifications_enabled = True
        if "cdr_reports" in missing_non_critical:
            st.session_state.cdr_enabled = False
        else:
            st.session_state.cdr_enabled = True
    else:
        st.session_state.notifications_enabled = True
        st.session_state.cdr_enabled = True
        st.sidebar.success("✅ Connected to database successfully")
    return True

def parse_duration(duration_str):
    try:
        match = re.match(r'(\d+)s', str(duration_str))
        if match:
            return int(match.group(1))
        return 0
    except Exception as e:
        st.error(f"Error parsing duration '{duration_str}': {str(e)}")
        return 0

def determine_call_direction(source):
    source_str = str(source).strip()
    if source_str in AGENT_EXTENSIONS:
        return "outbound"
    elif source_str.startswith("+"):
        return "inbound"
    elif source_str.startswith("8001"):
        return "queue"
    return "unknown"

def save_cdr_data(supabase, cdr_data):
    try:
        # Debug: Log user context
        user_id = st.session_state.get("supabase_user_id", "Unknown")
        user_data = supabase.table("users").select("name, role").eq("id", user_id).execute()
        st.write("**Debug: User Context for CDR Save**")
        st.write(f"User ID: {user_id}")
        st.write(f"User Data: {user_data.data if user_data.data else 'No user data found'}")
        
        for _, row in cdr_data.iterrows():
            agent_name = None
            source = str(row['Source']).strip()
            destination = str(row['Destination']).strip()
            if source in AGENT_EXTENSIONS:
                agent_name = AGENT_EXTENSIONS[source]
            elif destination in AGENT_EXTENSIONS and row['Status'].upper() == "ANSWERED":
                agent_name = AGENT_EXTENSIONS[destination]
            
            # Ensure agent_name is valid
            if agent_name and agent_name not in AGENT_EXTENSIONS.values():
                st.warning(f"Invalid agent_name '{agent_name}' for Uniqueid {row['Uniqueid']}. Setting to NULL.")
                agent_name = None
            
            cdr_entry = {
                "date": pd.to_datetime(row['Date']).isoformat(),
                "source": source,
                "ring_group": str(row['Ring Group']),
                "destination": destination,
                "src_channel": str(row['Src. Channel']),
                "account_code": str(row['Account Code']),
                "dst_channel": str(row['Dst. Channel']),
                "status": str(row['Status']),
                "duration": parse_duration(row['Duration']),
                "uniqueid": str(row['Uniqueid']),
                "user_field": str(row['User Field']),
                "call_direction": determine_call_direction(source),
                "agent_name": agent_name,
                "created_at": datetime.now().isoformat()
            }
            existing = supabase.table("cdr_reports").select("uniqueid").eq("uniqueid", cdr_entry["uniqueid"]).execute()
            if not existing.data:
                supabase.table("cdr_reports").insert(cdr_entry).execute()
            else:
                supabase.table("cdr_reports").update(cdr_entry).eq("uniqueid", cdr_entry["uniqueid"]).execute()
        return True
    except Exception as e:
        st.error(f"Error saving CDR data: {str(e)}")
        if "row-level security policy" in str(e):
            st.error("🔒 RLS policy is blocking the operation. Check user authentication and role in 'users' table.")
            st.write("Suggestions:")
            st.write("- Ensure you are logged in as a Manager with role='Manager' in the users table.")
            st.write("- Verify Supabase Authentication is enabled and the client uses a JWT token.")
            st.write("- Check RLS policies for 'cdr_reports' table in Supabase dashboard.")
        return False

def get_cdr_data(supabase, agent_name=None, start_date=None, end_date=None):
    try:
        query = supabase.table("cdr_reports").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        if start_date:
            query = query.gte("date", start_date.isoformat())
        if end_date:
            query = query.lte("date", end_date.isoformat())
        response = query.execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df['date'] = pd.to_datetime(df['date'])
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0).astype(int)
            return df
        st.warning(f"No CDR data found for agent: {agent_name or 'All'}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving CDR data: {str(e)}")
        if "violates row-level security policy" in str(e):
            st.error("🔒 RLS policy is blocking data access. Ensure agents are allowed to view their own data.")
        return pd.DataFrame()

def save_kpis(supabase, kpis):
    try:
        for metric, value in kpis.items():
            if value < 0:
                st.error(f"Invalid value for {metric}: Must be non-negative.")
                return False
            if metric == 'aht' and value == 0:
                st.error("Average Handle Time (AHT) cannot be zero.")
                return False
            if metric != 'call_volume' and value > 100 and metric not in ['aht', 'talk_time']:
                st.error(f"Invalid value for {metric}: Must be <= 100%.")
                return False
        existing_kpis = supabase.table("kpis").select("*").limit(1).execute()
        if existing_kpis.data:
            supabase.table("kpis").update(kpis).eq("id", existing_kpis.data[0]["id"]).execute()
        else:
            kpis["id"] = str(uuid.uuid4())
            supabase.table("kpis").insert(kpis).execute()
        return True
    except Exception as e:
        st.error(f"Error saving KPIs: {str(e)}")
        return False

def get_kpis(supabase):
    try:
        response = supabase.table("kpis").select("*").limit(1).execute()
        return response.data[0] if response.data else {}
    except Exception as e:
        st.error(f"Error retrieving KPIs: {str(e)}")
        return {}

def save_performance(supabase, agent_name, data):
    try:
        data["agent_name"] = agent_name
        data["date"] = datetime.now().isoformat()
        supabase.table("performance").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving performance for {agent_name}: {str(e)}")
        return False

def get_performance(supabase, agent_name=None):
    try:
        query = supabase.table("performance").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving performance: {str(e)}")
        return pd.DataFrame()

def assess_performance(df, kpis):
    results = df.copy()
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'resolution_rate', 'csat']
    results['overall_score'] = 0.0
    for metric in metrics:
        if metric in kpis:
            results[f'{metric}_pass'] = results[metric] >= kpis[metric]
            results['overall_score'] += results[f'{metric}_pass'].astype(int) * (100 / len(metrics))
    if 'call_volume' in kpis:
        results['call_volume_pass'] = results['call_volume'] >= kpis['call_volume']
        results['overall_score'] += results['call_volume_pass'].astype(int) * (100 / (len(metrics) + 1))
    if 'talk_time' in kpis:
        results['talk_time_pass'] = results['talk_time'] >= kpis['talk_time']
        results['overall_score'] += results['talk_time_pass'].astype(int) * (100 / (len(metrics) + 2))
    if 'aht' in kpis:
        results['aht_pass'] = results['aht'] <= kpis['aht']
        results['overall_score'] += results['aht_pass'].astype(int) * (100 / (len(metrics) + 3))
    return results

def set_agent_goal(supabase, agent_name, metric, target_value, created_by, is_manager=False):
    try:
        goal_data = {
            "agent_name": agent_name,
            "metric": metric,
            "target_value": target_value,
            "created_by": created_by,
            "status": "Approved" if is_manager else "Awaiting Approval",
            "created_at": datetime.now().isoformat()
        }
        supabase.table("goals").insert(goal_data).execute()
        if not is_manager and st.session_state.get("notifications_enabled", False):
            managers = supabase.table("users").select("id").eq("role", "Manager").execute()
            for manager in managers.data:
                supabase.table("notifications").insert({
                    "user_id": manager["id"],
                    "message": f"New goal for {agent_name} on {metric} awaiting approval."
                }).execute()
        return True
    except Exception as e:
        st.error(f"Error setting goal: {str(e)}")
        return False

def approve_goal(supabase, goal_id, manager_name, approve=True):
    try:
        schema_check = supabase.table("goals").select("approved_by").limit(1).execute()
        include_approved_by = 'approved_by' in schema_check.data[0] if schema_check.data else False
        update_data = {
            "status": "Approved" if approve else "Rejected",
            "approved_at": datetime.now().isoformat()
        }
        if include_approved_by:
            update_data["approved_by"] = manager_name
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

def get_feedback(supabase, agent_name=None):
    try:
        query = supabase.table("feedback").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving feedback: {str(e)}")
        return pd.DataFrame()

def respond_to_feedback(supabase, feedback_id, response, manager_name):
    try:
        update_data = {
            "manager_response": response,
            "response_timestamp": datetime.now().isoformat(),
            "updated_by": manager_name
        }
        supabase.table("feedback").update(update_data).eq("id", feedback_id).execute()
        if st.session_state.get("notifications_enabled", False):
            feedback = supabase.table("feedback").select("agent_name").eq("id", feedback_id).execute()
            if feedback.data:
                agent_name = feedback.data[0]["agent_name"]
                agent = supabase.table("users").select("id").eq("name", agent_name).execute()
                if agent.data:
                    supabase.table("notifications").insert({
                        "user_id": agent.data[0]["id"],
                        "message": f"New manager response to your feedback: {response[:50]}..."
                    }).execute()
        return True
    except Exception as e:
        st.error(f"Error responding to feedback: {str(e)}")
        return False

def get_notifications(supabase):
    try:
        user_id = supabase.table("users").select("id").eq("name", st.session_state.user).single().execute().data["id"]
        response = supabase.table("notifications").select("*").eq("user_id", user_id).eq("read", False).execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving notifications: {str(e)}")
        return pd.DataFrame()

def upload_audio(supabase, agent_name, audio_file, uploaded_by):
    try:
        file_name = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_file.name}"
        file_data = audio_file.read()
        supabase.storage.from_("audio_assessments").upload(file_name, file_data)
        audio_url = supabase.storage.from_("audio_assessments").get_public_url(file_name)
        supabase.table("audio_assessments").insert({
            "agent_name": agent_name,
            "audio_url": audio_url,
            "uploaded_by": uploaded_by,
            "upload_timestamp": datetime.now().isoformat(),
            "assessment_notes": ""
        }).execute()
        return True
    except Exception as e:
        st.error(f"Error uploading audio: {str(e)}")
        return False

def get_audio_assessments(supabase, agent_name=None):
    try:
        query = supabase.table("audio_assessments").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving audio assessments: {str(e)}")
        return pd.DataFrame()

def update_assessment_notes(supabase, assessment_id, notes):
    try:
        supabase.table("audio_assessments").update({"assessment_notes": notes}).eq("id", assessment_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating assessment notes: {str(e)}")
        return False

def get_zoho_agent_data(supabase, agent_name=None):
    try:
        chunk_size = 1000
        offset = 0
        all_data = []
        while True:
            query = supabase.table("zoho_agent_data").select("*").range(offset, offset + chunk_size - 1)
            if agent_name:
                query = query.eq("ticket_owner", agent_name)
            response = query.execute()
            if not response.data:
                break
            all_data.extend(response.data)
            offset += chunk_size
        if all_data:
            return pd.DataFrame(all_data)
        st.warning(f"No Zoho data found for agent: {agent_name or 'All'}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving Zoho data: {str(e)}")
        if "violates row-level security policy" in str(e):
            st.error("🔒 RLS policy is blocking data access. Ensure agents are allowed to view their own data.")
        return pd.DataFrame()

def setup_realtime(supabase):
    if st.session_state.get("auto_refresh", False):
        current_time = datetime.now()
        if (current_time - st.session_state.last_refresh).total_seconds() >= 30:
            st.session_state.data_updated = True
            st.session_state.last_refresh = current_time
            st.sidebar.success("Data refreshed!")

def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    st.markdown("""
        <style>
        .reportview-container {
            background: linear-gradient(to right, #f0f4f8, #e0e7ff);
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-right: 2px solid #4CAF50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #388E3C;
        }
        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Arial', sans-serif;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .progress-bar {
            height: 20px;
            border-radius: 5px;
        }
        .feedback-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            background-color: #e5ddd5;
            border-radius: 8px;
        }
        .feedback-item {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
        }
        .agent-msg {
            background-color: #dcf8c6;
            margin-left: auto;
            text-align: right;
        }
        .manager-msg {
            background-color: #fff;
            margin-right: auto;
        }
        .timestamp {
            font-size: 0.7em;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

    try:
        supabase = init_supabase()
        if not check_db(supabase):
            st.error("Critical database tables are missing. Please check the sidebar for details.")
            st.stop()
        global auth
        auth = supabase.auth
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        st.stop()

    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.supabase_user_id = None
        st.session_state.data_updated = False
        st.session_state.notifications_enabled = False
        st.session_state.cdr_enabled = False
        st.session_state.auto_refresh = False
        st.session_state.last_refresh = datetime.now()
        st.session_state.cleared_chats = set()

    if not st.session_state.user:
        st.title("🔐 Login")
        with st.form("login_form"):
            name = st.text_input("Name")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                success, user, role, user_id = authenticate_user(supabase, name, password)
                if success:
                    st.session_state.user = user
                    st.session_state.role = role
                    set_supabase_session(supabase, user_id)
                    st.success(f"Logged in as {user} ({role})")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        return

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.supabase_user_id = None
        supabase.auth.sign_out()
        st.rerun()

    if st.session_state.get("notifications_enabled", False):
        notifications = get_notifications(supabase)
        with st.sidebar.expander(f"🔔 Notifications ({len(notifications)})"):
            if notifications.empty:
                st.write("No new notifications.")
            else:
                for _, notif in notifications.iterrows():
                    st.write(notif["message"])
                    if st.button("Mark as Read", key=f"notif_{notif['id']}"):
                        supabase.table("notifications").update({"read": True}).eq("id", notif["id"]).execute()
                        st.rerun()
    else:
        with st.sidebar.expander("🔔 Notifications (0)"):
            st.write("Notifications disabled (notifications table missing).")

    st.session_state.auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    setup_realtime(supabase)
    if st.session_state.get("auto_refresh", False) and st.session_state.get("data_updated", False):
        st.session_state.data_updated = False
        st.rerun()

    st.sidebar.info(f"👤 Logged in as: {st.session_state.user}")
    st.sidebar.info(f"🎓 Role: {st.session_state.role}")

    try:
        st.image(r"./companylogo.png", width=150)
    except Exception as e:
        st.warning(f"Failed to load company logo: {str(e)}")

    if st.session_state.role == "Manager":
        st.title("📊 Manager Dashboard")
        performance_df = get_performance(supabase)
        if not performance_df.empty:
            kpis = get_kpis(supabase)
            results = assess_performance(performance_df, kpis)
            avg_overall_score = results['overall_score'].mean()
            total_call_volume = performance_df['call_volume'].sum()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Overall Score", f"{avg_overall_score:.1f}%")
            with col2:
                st.metric("Total Call Volume", f"{total_call_volume}")
            with col3:
                st.metric("Agent Count", len(results['agent_name'].unique()))
        tabs = st.tabs(["📋 Set KPIs", "📝 Input Performance", "📊 Assessments", "🎯 Set Goals", "💬 Feedback", "🎙️ Audio Assessments", "📞 CDR Reports"])

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
                            st.success(f"Performance saved for {agent}!")
            
            st.subheader("Upload Performance Data")
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                required_cols = ['agent_name', 'attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                                'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']
                if all(col in df.columns for col in required_cols):
                    for _, row in df.iterrows():
                        data = {col: row[col] for col in required_cols[1:]}
                        if 'date' in row:
                            data['date'] = row['date']
                        save_performance(supabase, row['agent_name'], data)
                    st.success(f"Imported data for {len(df)} agents!")
                else:
                    st.error("CSV missing required columns.")

        with tabs[2]:
            st.header("📊 Assessment Results")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=pd.to_datetime('2025-05-01'))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now().date())
            if not performance_df.empty:
                performance_df['date'] = pd.to_datetime(performance_df['date'])
                masked_df = performance_df[(performance_df['date'] >= pd.to_datetime(start_date)) & 
                                        (performance_df['date'] <= pd.to_datetime(end_date))]
                kpis = get_kpis(supabase)
                results = assess_performance(masked_df, kpis)
                st.dataframe(results)
                st.download_button(label="📥 Download Data", data=results.to_csv(index=False), file_name="performance_data.csv")
                try:
                    fig = px.bar(results, x='agent_name', y='overall_score', color='agent_name', 
                                title="Agent Overall Scores", labels={'overall_score': 'Score (%)'})
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error plotting data: {str(e)}")

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
                            if 'created_by' in row:
                                st.write(f"Created by: {row['created_by']}")
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
                    if 'created_by' in goals_display_df.columns:
                        display_columns.insert(4, 'created_by')
                    if 'approved_by' in goals_display_df.columns:
                        display_columns.append('approved_by')
                    if 'approved_at' in goals_display_df.columns:
                        display_columns.append('approved_at')
                    st.dataframe(goals_display_df[display_columns])
                    st.download_button(label="📥 Download Goals", data=goals_display_df.to_csv(index=False), file_name="agent_goals.csv")
                else:
                    st.info("No goals set.")

        with tabs[4]:
            st.header("💬 View and Respond to Agent Feedback")
            feedback_df = get_feedback(supabase)
            show_debug = st.checkbox("Show Debug Info", key="feedback_debug")
            if show_debug:
                st.write("Debug: Session State", st.session_state)
                st.write("Debug: Feedback Data", feedback_df)
            if not feedback_df.empty:
                feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                feedback_df['response_timestamp'] = pd.to_datetime(feedback_df['response_timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                display_columns = ['agent_name', 'message', 'created_at', 'manager_response', 'response_timestamp']
                if 'updated_by' in feedback_df.columns:
                    display_columns.append('updated_by')
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
                            st.success("Response sent!")
                            if 'reply_to_feedback_id' in st.session_state:
                                del st.session_state.reply_to_feedback_id
                            st.rerun()
                        else:
                            st.error("Failed to send response.")
                    elif submit:
                        st.error("Please provide a response and ensure a feedback is selected.")
            else:
                st.info("No feedback submitted.")

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
                                st.success("Notes saved!")
                                st.rerun()
                            else:
                                st.error("Failed to save notes.")
                st.dataframe(audio_df[['agent_name', 'upload_timestamp', 'uploaded_by', 'assessment_notes']])
                st.download_button(label="📥 Download Audio Assessments", data=audio_df.to_csv(index=False), file_name="audio_assessments.csv")
            else:
                st.info("No audio assessments available.")

        with tabs[6]:
            st.header("📞 CDR Reports")
            if not st.session_state.get("cdr_enabled", False):
                st.warning("CDR functionality is disabled because the 'cdr_reports' table is missing.")
            else:
                st.subheader("Upload CDR Data")
                uploaded_file = st.file_uploader("Upload CDR CSV", type="csv", key="cdr_upload")
                if uploaded_file:
                    try:
                        # Read CSV with explicit encoding and flexible delimiter
                        df = pd.read_csv(uploaded_file, encoding='utf-8-sig', sep=',', on_bad_lines='warn')
                        
                        # Debug: Show original column names
                        st.write("**Debug: CSV Column Names**")
                        st.write(df.columns.tolist())
                        
                        # Normalize column names: strip whitespace, handle case sensitivity
                        df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
                        column_mappings = {
                            'uniqueid': 'Uniqueid',
                            'unique id': 'Uniqueid',
                            'UniqueID': 'Uniqueid',
                            'Unique ID': 'Uniqueid'
                        }
                        df.columns = [column_mappings.get(col.lower(), col) for col in df.columns]
                        
                        # Required columns
                        required_cols = ['Date', 'Source', 'Ring Group', 'Destination', 'Src. Channel', 
                                       'Account Code', 'Dst. Channel', 'Status', 'Duration', 'Uniqueid', 'User Field']
                        
                        # Check for missing columns
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            st.error(f"CSV missing required columns: {', '.join(missing_cols)}")
                            st.write("Expected columns:", required_cols)
                            st.write("Current columns after normalization:", df.columns.tolist())
                            st.write("Please ensure column names match exactly (check for case sensitivity or extra spaces).")
                        else:
                            # Select only required columns
                            df = df[required_cols]
                            
                            # Debug: Confirm selected columns
                            st.write("**Debug: Selected Columns for Processing**")
                            st.write(df.columns.tolist())
                            
                            try:
                                df['Date'] = pd.to_datetime(df['Date'])
                                df['Duration'] = df['Duration'].apply(parse_duration)
                                if df['Duration'].lt(0).any():
                                    st.error("Duration cannot be negative.")
                                elif df['Uniqueid'].duplicated().any():
                                    st.error("Duplicate Uniqueid values found.")
                                else:
                                    if save_cdr_data(supabase, df):
                                        st.success(f"Imported CDR data for {len(df)} records!")
                                    else:
                                        st.error("Failed to import CDR data.")
                            except Exception as e:
                                st.error(f"Invalid data format: {str(e)}")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        st.write("Try checking the CSV for correct delimiter (comma), encoding (UTF-8), or malformed rows.")

                st.subheader("View CDR Data")
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=pd.to_datetime('2025-05-01'), key="cdr_start")
                with col2:
                    end_date = st.date_input("End Date", value=datetime.now().date(), key="cdr_end")
                cdr_df = get_cdr_data(supabase, start_date=start_date, end_date=end_date)
                if not cdr_df.empty:
                    st.dataframe(cdr_df[['date', 'source', 'ring_group', 'destination', 'src_channel', 'account_code', 'dst_channel', 'status', 'duration', 'uniqueid', 'user_field', 'call_direction', 'agent_name']])
                    st.download_button(label="📥 Download CDR Data", data=cdr_df.to_csv(index=False), file_name="cdr_reports.csv")
                    try:
                        direction_counts = cdr_df.groupby('call_direction')['uniqueid'].nunique().reset_index(name='Count')
                        fig1 = px.pie(direction_counts, values='Count', names='call_direction', title="Call Direction Distribution")
                        st.plotly_chart(fig1)
                        status_counts = cdr_df.groupby('status')['uniqueid'].nunique().reset_index(name='Count')
                        fig2 = px.bar(status_counts, x='status', y='Count', title="Call Status Distribution")
                        st.plotly_chart(fig2)
                    except Exception as e:
                        st.error(f"Error plotting CDR data: {str(e)}")
                else:
                    st.info("No CDR data available for the selected date range.")

    elif st.session_state.role == "Agent":
        st.title(f"👤 Agent Dashboard - {st.session_state.user}")
        if st.session_state.user == "Joseph Kavuma":
            try:
                st.image("Joseph.jpg", caption="Agent Profile", width=150)
            except:
                st.error("Error loading profile image.")
        
        tabs = st.tabs(["📋 Metrics", "🎯 Goals", "💬 Feedback", "📊 Tickets", "📞 CDR Reports"])
        performance_df = get_performance(supabase, st.session_state.user)
        all_performance_df = get_performance(supabase)
        zoho_df = get_zoho_agent_data(supabase, st.session_state.user)

        with tabs[0]:
            with st.expander("📈 Performance Metrics"):
                if not performance_df.empty and not all_performance_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Start Date", value=pd.to_datetime('2025-05-01'), key="agent_start")
                    with col2:
                        end_date = st.date_input("End Date", value=datetime.now().date(), key="agent_end")
                    performance_df['date'] = pd.to_datetime(performance_df['date'])
                    all_performance_df['date'] = pd.to_datetime(all_performance_df['date'])
                    masked_df = performance_df[(performance_df['date'] >= pd.to_datetime(start_date)) & 
                                             (performance_df['date'] <= pd.to_datetime(end_date))]
                    all_masked_df = all_performance_df[(all_performance_df['date'] >= pd.to_datetime(start_date)) & 
                                                     (all_performance_df['date'] <= pd.to_datetime(end_date))]
                    kpis = get_kpis(supabase)
                    results = assess_performance(masked_df, kpis)
                    all_results = assess_performance(all_masked_df, kpis)
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
                    metrics = ['quality_score', 'csat', 'attendance', 'resolution_rate']
                    values = [results[m].mean() for m in metrics]
                    fig = go.Figure(data=go.Scatterpolar(r=values, theta=[m.replace('_', ' ').title() for m in metrics], fill='toself'))
                    fig.update_layout(title="Your Performance Profile", polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
                    st.plotly_chart(fig)
                    
                    st.subheader("Comparison to Peers")
                    peer_avg = all_results.groupby('agent_name')['overall_score'].mean().reset_index()
                    peer_avg = peer_avg[peer_avg['agent_name'] != st.session_state.user]
                    fig3 = px.box(peer_avg, y='overall_score', title="Peer Score Distribution", labels={'overall_score': 'Score (%)'}, points="all")
                    fig3.add_hline(y=avg_overall_score, line_dash="dash", line_color="red", annotation_text=f"Your Score: {avg_overall_score:.1f}%")
                    st.plotly_chart(fig3)
                else:
                    st.info("No performance data available.")

        with tabs[1]:
            with st.expander("🎯 Your Goals"):
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
                            # Calculate progress
                            if metric == 'aht':
                                kpi_target = kpis.get(metric, 600)
                                if kpi_target > 0:
                                    progress = ((kpi_target - current_value) / kpi_target - row['target_value']) * 100
                                else:
                                    progress = 0
                                    st.error("KPI target for AHT is zero, cannot calculate progress.")
                                    supabase.table("error_logs").insert({
                                        "message": f"AHT KPI target is zero for agent {st.session_state.user}",
                                        "timestamp": datetime.now().isoformat()
                                    }).execute()
                                progress = min(max(progress, 0), 100)
                            else:
                                if row['target_value'] > 0:
                                    progress = (current_value / row['target_value']) * 100
                                else:
                                    progress = 0
                                progress = min(max(progress, 0), 100)
                            
                            color = "green" if progress >= 80 else "orange" if progress >= 50 else "red"
                            st.markdown(f"<div class='progress-bar' style='background-color: {color}; width: {progress}%;'></div>", unsafe_allow_html=True)
                            st.write(f"{metric.replace('_', ' ').title()}: Target {row['target_value']:.1f}{' sec' if metric == 'aht' else ''}, Current {current_value:.1f}{' sec' if metric == 'aht' else ''}, Status: {row['status']}")
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

        with tabs[2]:
            with st.expander("💬 Feedback and Responses"):
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
                        st.success("Feedback submitted!")
                
                st.write("**Feedback History**")
                feedback_df = get_feedback(supabase, st.session_state.user)
                if not feedback_df.empty:
                    feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    feedback_df['response_timestamp'] = pd.to_datetime(feedback_df['response_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    display_columns = ['message', 'created_at', 'manager_response', 'response_timestamp']
                    if 'updated_by' in feedback_df.columns:
                        display_columns.append('updated_by')
                    st.dataframe(feedback_df[display_columns])
                    st.download_button(label="📥 Download Feedback", data=feedback_df.to_csv(index=False), file_name="feedback_history.csv")
                else:
                    st.info("No feedback submitted.")

        with tabs[3]:
            with st.expander("📊 Zoho Ticket Data"):
                if not zoho_df.empty:
                    total_tickets = zoho_df['id'].nunique()
                    st.metric("Total Tickets Handled", f"{total_tickets}")
                    show_debug = st.checkbox("Show Debug: Raw Zoho Data")
                    if show_debug:
                        st.write(f"Logged-in user: {st.session_state.user}")
                        st.write(f"Unique ticket_owner values: {zoho_df['ticket_owner'].unique()}")
                        st.write(f"Total rows: {len(zoho_df)}")
                        st.write(f"Unique ticket IDs: {zoho_df['id'].nunique()}")
                        st.dataframe(zoho_df)
                    channel_counts = zoho_df.groupby('channel')['id'].nunique().reset_index(name='Ticket Count')
                    st.write("**Ticket Breakdown by Channel**")
                    st.dataframe(channel_counts)
                    try:
                        fig = px.pie(channel_counts, values='Ticket Count', names='channel', title="Ticket Distribution by Channel")
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error plotting: {str(e)}")
                    st.download_button(label="📥 Download Zoho Data", data=zoho_df.to_csv(index=False), file_name="zoho_agent_data.csv")
                else:
                    st.info("No Zoho data available.")
                    st.write("Debug: Check zoho_agent_data table and RLS policies.")

        with tabs[4]:
            with st.expander("📞 Your CDR Data"):
                if not st.session_state.get("cdr_enabled", False):
                    st.warning("CDR data is unavailable because the 'cdr_reports' table is missing.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input("Start Date", value=pd.to_datetime('2025-05-01'), key="agent_cdr_start")
                    with col2:
                        end_date = st.date_input("End Date", value=datetime.now().date(), key="agent_cdr_end")
                    cdr_df = get_cdr_data(supabase, agent_name=st.session_state.user, start_date=start_date, end_date=end_date)
                    if not cdr_df.empty:
                        total_calls = cdr_df['uniqueid'].nunique()
                        avg_duration = cdr_df['duration'].mean()
                        answered_calls = cdr_df[(cdr_df['status'] == 'ANSWERED') & (cdr_df['destination'].isin(AGENT_EXTENSIONS))]['uniqueid'].nunique()
                        queue_calls = cdr_df[cdr_df['call_direction'] == 'queue']['uniqueid'].nunique()
                        st.metric("Total Calls", f"{total_calls}")
                        st.metric("Answered Calls", f"{answered_calls}")
                        st.metric("Queue Calls", f"{queue_calls}")
                        st.metric("Average Call Duration", f"{avg_duration:.1f} seconds")
                        st.dataframe(cdr_df[['date', 'source', 'ring_group', 'destination', 'status', 'duration', 'call_direction', 'user_field']])
                        st.download_button(label="📥 Download Your CDR Data", data=cdr_df.to_csv(index=False), file_name="my_cdr_data.csv")
                        try:
                            direction_counts = cdr_df.groupby('call_direction')['uniqueid'].nunique().reset_index(name='Count')
                            fig = px.bar(direction_counts, x='call_direction', y='Count', title="Your Call Direction Distribution")
                            st.plotly_chart(fig)
                        except Exception as e:
                            st.error(f"Error plotting CDR data: {str(e)}")
                    else:
                        st.info("No CDR data available for the selected date range.")

if __name__ == "__main__":
    main()
