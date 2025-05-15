import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client
import uuid

def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        if not url.startswith("https://"):
            url = f"https://{url}"
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        raise e

def check_db(supabase):
    required_tables = ["users", "kpis", "performance", "zoho_agent_data", "goals", "feedback", "notifications", "audio_assessments"]
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
    else:
        st.session_state.notifications_enabled = True
        st.sidebar.success("✅ Connected to database successfully")
    return True

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

def get_kpis(supabase):
    try:
        response = supabase.table("kpis").select("*").execute()
        kpis = {}
        for row in response.data:
            metric = row["metric"]
            value = row["threshold"]
            kpis[metric] = int(float(value)) if metric == "call_volume" else float(value) if value is not None else 0.0
        return kpis
    except Exception as e:
        st.error(f"Error retrieving KPIs: {str(e)}")
        return {}

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
        update_goal_status(supabase, agent_name)
        return True
    except Exception as e:
        st.error(f"Error saving performance data: {str(e)}")
        return False

def get_performance(supabase, agent_name=None):
    try:
        query = supabase.table("performance").select("*")
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

def get_zoho_agent_data(supabase, agent_name=None, start_date=None, end_date=None):
    try:
        all_data = []
        chunk_size = 1000
        offset = 0

        while True:
            query = supabase.table("zoho_agent_data").select("*").range(offset, offset + chunk_size - 1)
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
            if 'id' not in df.columns:
                st.error("❌ The 'zoho_agent_data' table is missing an 'id' column.")
                return pd.DataFrame()
            if 'ticket_owner' not in df.columns:
                st.error("❌ The 'zoho_agent_data' table is missing a 'ticket_owner' column.")
                return pd.DataFrame()
            st.write(f"✅ Supabase returned {len(df)} rows for agent: {agent_name or 'All'}")
            return df
        else:
            st.warning(f"⚠️ No Zoho agent data found for agent '{agent_name}'.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error retrieving Zoho agent data: {str(e)}")
        return pd.DataFrame()

def set_agent_goal(supabase, agent_name, metric, target_value, manager_name, approval_status="Approved"):
    try:
        schema_check = supabase.table("goals").select("created_by").limit(1).execute()
        include_created_by = 'created_by' in schema_check.data[0] if schema_check.data else False
        goal_data = {
            "agent_name": agent_name,
            "metric": metric,
            "target_value": target_value,
            "status": "Pending",
            "approval_status": approval_status
        }
        if include_created_by:
            goal_data["created_by"] = manager_name
        response = supabase.table("goals").select("*").eq("agent_name", agent_name).eq("metric", metric).execute()
        if response.data:
            supabase.table("goals").update(goal_data).eq("agent_name", agent_name).eq("metric", metric).execute()
        else:
            supabase.table("goals").insert(goal_data).execute()
        return True
    except Exception as e:
        st.error(f"Error setting goal: {str(e)}")
        return False

def approve_goal(supabase, goal_id, manager_name, approve=True):
    try:
        schema_check = supabase.table("goals").select("updated_by").limit(1).execute()
        include_updated_by = 'updated_by' in schema_check.data[0] if schema_check.data else False
        update_data = {
            "approval_status": "Approved" if approve else "Rejected",
            "updated_at": datetime.now().isoformat()
        }
        if include_updated_by:
            update_data["updated_by"] = manager_name
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
                        "message": f"Your goal update was {status} by {manager_name}"
                    }).execute()
        return True
    except Exception as e:
        st.error(f"Error approving goal: {str(e)}")
        return False

def update_goal_status(supabase, agent_name):
    try:
        goals = supabase.table("goals").select("*").eq("agent_name", agent_name).eq("approval_status", "Approved").execute()
        perf = get_performance(supabase, agent_name)
        if not goals.data or perf.empty:
            return
        latest_perf = perf[perf['date'] == perf['date'].max()]
        for goal in goals.data:
            metric = goal['metric']
            target = goal['target_value']
            if metric in latest_perf.columns:
                value = latest_perf[metric].iloc[0]
                status = "Completed" if (metric == "aht" and value <= target) or (metric != "aht" and value >= target) else "Pending"
                supabase.table("goals").update({"status": status}).eq("id", goal['id']).execute()
    except Exception as e:
        st.error(f"Error updating goal status: {str(e)}")

def get_feedback(supabase, agent_name=None):
    try:
        query = supabase.table("feedback").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        st.warning(f"No feedback for {agent_name or 'any agents'}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving feedback: {str(e)}")
        return pd.DataFrame()

def respond_to_feedback(supabase, feedback_id, manager_response, manager_name):
    try:
        schema_check = supabase.table("feedback").select("updated_by").limit(1).execute()
        include_updated_by = 'updated_by' in schema_check.data[0] if schema_check.data else False
        response_data = {
            "manager_response": manager_response,
            "response_timestamp": datetime.now().isoformat()
        }
        if include_updated_by:
            response_data["updated_by"] = manager_name
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

def get_notifications(supabase):
    if not st.session_state.get("notifications_enabled", False):
        return pd.DataFrame()
    try:
        user_response = supabase.table("users").select("id").eq("name", st.session_state.user).execute()
        if not user_response.data:
            st.warning("User not found in users table.")
            return pd.DataFrame()
        user_id = user_response.data[0]["id"]
        response = supabase.table("notifications").select("*").eq("user_id", user_id).eq("read", False).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving notifications: {str(e)}")
        return pd.DataFrame()

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

def authenticate_user(supabase, name, password):
    try:
        user_response = supabase.table("users").select("*").eq("name", name).execute()
        if user_response.data:
            return True, name, user_response.data[0]["role"]
        return False, None, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None, None

def setup_realtime(supabase):
    if st.session_state.get("auto_refresh", False):
        current_time = datetime.now()
        last_refresh = st.session_state.get("last_refresh", current_time)
        if current_time - last_refresh >= timedelta(seconds=30):
            st.session_state.data_updated = True
            st.session_state.last_refresh = current_time
        st.sidebar.success("Auto-refresh enabled (polling every 30 seconds).")
    else:
        st.sidebar.info("Auto-refresh disabled. Enable to poll data every 30 seconds.")

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

def get_audio_assessments(supabase, agent_name=None):
    try:
        query = supabase.table("audio_assessments").select("*")
        if agent_name:
            query = query.eq("agent_name", agent_name)
        response = query.execute()
        if response.data:
            return pd.DataFrame(response.data)
        st.warning(f"No audio assessments for {agent_name or 'any agents'}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving audio assessments: {str(e)}")
        return pd.DataFrame()

def update_assessment_notes(supabase, audio_id, notes):
    try:
        supabase.table("audio_assessments").update({"assessment_notes": notes}).eq("id", audio_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating assessment notes: {str(e)}")
        return False

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
        st.session_state.data_updated = False
        st.session_state.notifications_enabled = False
        st.session_state.auto_refresh = False
        st.session_state.last_refresh = datetime.now()
        st.session_state.cleared_chats = set()

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
                    st.success(f"Logged in as {user} ({role})")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        return

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.role = None
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
        tabs = st.tabs(["📋 Set KPIs", "📝 Input Performance", "📊 Assessments", "🎯 Set Goals", "💬 Feedback", "🎙️ Audio Assessments", "✅ Approve Goals"])

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
                    onboarding = st.number_input("Onboarding (%)", min_value=0.0, max_value=100.0, step=0.1)
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
                        if set_agent_goal(supabase, agent, metric, target_value, st.session_state.user):
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
                            set_agent_goal(supabase, agent, bulk_metric, bulk_target, st.session_state.user)
                        st.success(f"Goals set for {len(bulk_agents)} agents!")

                st.subheader("Current Goals")
                goals_df = supabase.table("goals").select("*").in_("agent_name", agents).execute()
                if goals_df.data:
                    goals_display_df = pd.DataFrame(goals_df.data)
                    goals_display_df['target_value'] = goals_display_df.apply(
                        lambda x: f"{x['target_value']:.1f}{' sec' if x['metric'] == 'aht' else ''}", axis=1)
                    display_columns = ['agent_name', 'metric', 'target_value', 'status', 'approval_status', 'created_at']
                    if 'created_by' in goals_display_df.columns:
                        display_columns.insert(4, 'created_by')
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
            st.header("✅ Approve Agent Goal Updates")
            pending_goals = supabase.table("goals").select("*").eq("approval_status", "Pending").execute()
            if pending_goals.data:
                pending_df = pd.DataFrame(pending_goals.data)
                pending_df['target_value'] = pending_df.apply(
                    lambda x: f"{x['target_value']:.1f}{' sec' if x['metric'] == 'aht' else ''}", axis=1)
                display_columns = ['agent_name', 'metric', 'target_value', 'status', 'approval_status', 'created_at']
                if 'created_by' in pending_df.columns:
                    display_columns.insert(4, 'created_by')
                st.dataframe(pending_df[display_columns])
                
                for _, row in pending_df.iterrows():
                    with st.expander(f"Review: {row['agent_name']} - {row['metric']}"):
                        st.write(f"Target Value: {row['target_value']}")
                        st.write(f"Status: {row['status']}")
                        st.write(f"Approval Status: {row['approval_status']}")
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
                st.info("No pending goal updates to approve.")

    elif st.session_state.role == "Agent":
        st.title(f"👤 Agent Dashboard - {st.session_state.user}")
        if st.session_state.user == "Joseph Kavuma":
            try:
                st.image("Joseph.jpg", caption="Agent Profile", width=150)
            except:
                st.error("Error loading profile image.")
        
        tabs = st.tabs(["📋 Metrics", "🎯 Goals", "💬 Feedback", "📊 Tickets"])
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
                            progress = min((kpis.get(metric, 600) - current_value) / (kpis.get(metric, 600) - row['target_value']) * 100, 100) if metric == 'aht' else min(current_value / row['target_value'] * 100, 100) if row['target_value'] > 0 else 0
                            color = "green" if progress >= 80 else "orange" if progress >= 50 else "red"
                            st.markdown(f"<div class='progress-bar' style='background-color: {color}; width: {progress}%;'></div>", unsafe_allow_html=True)
                            st.write(f"{metric.replace('_', ' ').title()}: Target {row['target_value']:.1f}{' sec' if metric == 'aht' else '%'}, Current {current_value:.1f}{' sec' if metric == 'aht' else '%'}, Status: {row['status']}, Approval: {row['approval_status']}")
                            if row['approval_status'] != "Rejected":
                                with st.form(f"update_goal_form_{metric}"):
                                    new_target = st.number_input(f"New Target for {metric}", value=float(row['target_value']), min_value=0.0)
                                    if st.form_submit_button(f"Update {metric} Goal"):
                                        if set_agent_goal(supabase, st.session_state.user, metric, new_target, st.session_state.user, approval_status="Pending"):
                                            if st.session_state.get("notifications_enabled", False):
                                                managers = supabase.table("users").select("id").eq("role", "Manager").execute()
                                                for manager in managers.data:
                                                    supabase.table("notifications").insert({
                                                        "user_id": manager["id"],
                                                        "message": f"{st.session_state.user} requested to update {metric} goal to {new_target}"
                                                    }).execute()
                                            st.success("Goal update submitted! (Pending manager approval)")
                        else:
                            st.write(f"No goal set for {metric.replace('_', ' ').title()}.")
                else:
                    st.info("No goals set.")

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

if __name__ == "__main__":
    main()