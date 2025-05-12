import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from supabase import create_client, Client

# Supabase initialization
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

# Initialize database tables if they don't exist
def init_db(supabase):
    # Since Supabase uses PostgreSQL, tables need to be created via SQL migrations
    # or through the Supabase dashboard. This function is kept for consistency
    # but actual table creation should be done in the Supabase dashboard.
    
    # Insert default users if they don't exist already
    default_users = [
        {'email': 'tutumelchizedek8@gmail.com', 'role': 'Manager'},
        {'email': 'pammirembe@gmail.com', 'role': 'Manager'},
        {'email': 'daisynahabwe12@gmail.com', 'role': 'Agent'},
        {'email': 'tutu.melchizedek@bodabodaunion.ug', 'role': 'Agent'},
        {'email': 'josephkavuma606@gmail.com', 'role': 'Agent'},
        {'email': 'kyomarobert74@gmail.com', 'role': 'Agent'},
        {'email': 'amuletk95@gmail.com', 'role': 'Agent'},
        {'email': 'lunkusecynthia2@gmail.com', 'role': 'Agent'},
        {'email': 'jacobhum905@gmail.com', 'role': 'Agent'}
    ]
    
    for user in default_users:
        # Check if user exists
        response = supabase.table("users").select("*").eq("email", user["email"]).execute()
        if len(response.data) == 0:
            # User doesn't exist, insert
            supabase.table("users").insert(user).execute()

def get_db_connection():
    return init_supabase()

# Save KPIs
def save_kpis(supabase, kpis):
    for metric, threshold in kpis.items():
        # Check if KPI exists
        response = supabase.table("kpis").select("*").eq("metric", metric).execute()
        if len(response.data) == 0:
            # KPI doesn't exist, insert
            supabase.table("kpis").insert({"metric": metric, "threshold": threshold}).execute()
        else:
            # KPI exists, update
            supabase.table("kpis").update({"threshold": threshold}).eq("metric", metric).execute()

# Get KPIs
def get_kpis(supabase):
    response = supabase.table("kpis").select("*").execute()
    kpis = {row["metric"]: row["threshold"] for row in response.data}
    return kpis

# Save performance data
def save_performance(supabase, agent_email, data):
    date = datetime.now().strftime("%Y-%m-%d")
    performance_data = {
        "agent_email": agent_email,
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

# Get performance data
def get_performance(supabase, agent_email=None):
    if agent_email:
        response = supabase.table("performance").select("*").eq("agent_email", agent_email).execute()
    else:
        response = supabase.table("performance").select("*").execute()
    
    if response.data:
        return pd.DataFrame(response.data)
    else:
        return pd.DataFrame()

# Assess performance based on KPIs
def assess_performance(performance_df, kpis):
    if performance_df.empty:
        return performance_df
        
    results = performance_df.copy()
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'csat', 'call_volume']
    
    for metric in metrics:
        if metric in results.columns:
            if metric == 'aht':
                results[f'{metric}_pass'] = results[metric] <= kpis.get(metric, 600)
            else:
                results[f'{metric}_pass'] = results[metric] >= kpis.get(metric, 50)
    
    # Only calculate if all required columns exist
    pass_columns = [f'{m}_pass' for m in metrics if f'{m}_pass' in results.columns]
    if pass_columns:
        results['overall_score'] = results[pass_columns].mean(axis=1) * 100
    
    return results

# Streamlit app
def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    
    # Initialize Supabase client
    try:
        supabase = init_supabase()
        init_db(supabase)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return

    # Session state for authentication
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None

    # Auth via Supabase Auth UI (redirected back to this app)
    if not st.session_state.user:
        st.title("Login")
        
        # Simple login form (you can expand this to use Supabase Auth UI)
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                try:
                    # This is a simplified version. In a real app, you would use Supabase Auth
                    # Here we're just checking if the user exists in our users table
                    response = supabase.table("users").select("*").eq("email", email).execute()
                    if response.data:
                        st.session_state.user = email
                        st.session_state.role = response.data[0]["role"]
                        st.success(f"Logged in as {email} ({st.session_state.role})")
                        st.rerun()
                    else:
                        st.error("User not found or incorrect credentials")
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
        
        # Note about Supabase Auth
        st.info("Note: For production, you should use Supabase Authentication which provides secure user management.")
        return

    # Logout button
    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()

    # Manager interface
    if st.session_state.role == "Manager":
        st.title("Manager Dashboard")
        tabs = st.tabs(["Set KPIs", "Input Performance", "View Assessments"])

        # Set KPIs
        with tabs[0]:
            st.header("Set KPI Thresholds")
            kpis = get_kpis(supabase)
            with st.form("kpi_form"):
                attendance = st.number_input("Attendance (%, min)", value=kpis.get('attendance', 95.0), min_value=0.0, max_value=100.0)
                quality_score = st.number_input("Quality Score (%, min)", value=kpis.get('quality_score', 90.0), min_value=0.0, max_value=100.0)
                product_knowledge = st.number_input("Product Knowledge (%, min)", value=kpis.get('product_knowledge', 85.0), min_value=0.0, max_value=100.0)
                contact_success_rate = st.number_input("Contact Success Rate (%, min)", value=kpis.get('contact_success_rate', 80.0), min_value=0.0, max_value=100.0)
                onboarding = st.number_input("Onboarding (%, min)", value=kpis.get('onboarding', 90.0), min_value=0.0, max_value=100.0)
                reporting = st.number_input("Reporting (%, min)", value=kpis.get('reporting', 95.0), min_value=0.0, max_value=100.0)
                talk_time = st.number_input("CRM Talk Time (seconds, min)", value=kpis.get('talk_time', 300.0), min_value=0.0)
                resolution_rate = st.number_input("Issue Resolution Rate (%, min)", value=kpis.get('resolution_rate', 80.0), min_value=0.0, max_value=100.0)
                aht = st.number_input("Average Handle Time (seconds, max)", value=kpis.get('aht', 600.0), min_value=0.0)
                csat = st.number_input("Customer Satisfaction (%, min)", value=kpis.get('csat', 85.0), min_value=0.0, max_value=100.0)
                call_volume = st.number_input("Call Volume (calls, min)", value=kpis.get('call_volume', 50), min_value=0)
                if st.form_submit_button("Save KPIs"):
                    new_kpis = {
                        'attendance': attendance,
                        'quality_score': quality_score,
                        'product_knowledge': product_knowledge,
                        'contact_success_rate': contact_success_rate,
                        'onboarding': onboarding,
                        'reporting': reporting,
                        'talk_time': talk_time,
                        'resolution_rate': resolution_rate,
                        'aht': aht,
                        'csat': csat,
                        'call_volume': call_volume
                    }
                    save_kpis(supabase, new_kpis)
                    st.success("KPIs saved!")

        # Input Performance
        with tabs[1]:
            st.header("Input Agent Performance")
            # Get agents
            response = supabase.table("users").select("*").eq("role", "Agent").execute()
            agents = [user["email"] for user in response.data]
            
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
                        'attendance': attendance,
                        'quality_score': quality_score,
                        'product_knowledge': product_knowledge,
                        'contact_success_rate': contact_success_rate,
                        'onboarding': onboarding,
                        'reporting': reporting,
                        'talk_time': talk_time,
                        'resolution_rate': resolution_rate,
                        'aht': aht,
                        'csat': csat,
                        'call_volume': call_volume
                    }
                    save_performance(supabase, agent, data)
                    st.success("Performance data saved!")

        # View Assessments
        with tabs[2]:
            st.header("Assessment Results")
            performance_df = get_performance(supabase)
            if not performance_df.empty:
                kpis = get_kpis(supabase)
                results = assess_performance(performance_df, kpis)
                st.dataframe(results)
                st.subheader("Performance Overview")
                fig = px.bar(results, x='agent_email', y='overall_score', color='agent_email', 
                             title="Agent Overall Scores", labels={'overall_score': 'Score (%)'})
                st.plotly_chart(fig)
            else:
                st.write("No performance data available.")

    # Agent interface
    elif st.session_state.role == "Agent":
        st.title(f"Agent Dashboard - {st.session_state.user}")
        performance_df = get_performance(supabase, st.session_state.user)
        if not performance_df.empty:
            kpis = get_kpis(supabase)
            results = assess_performance(performance_df, kpis)
            st.dataframe(results)
            st.subheader("Your Performance")
            try:
                fig = px.line(results, x='date', y='overall_score', title="Your Score Over Time", 
                            labels={'overall_score': 'Score (%)'})
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error plotting data: {str(e)}")
                st.write("Raw data:")
                st.write(results)
        else:
            st.write("No performance data available.")

if __name__ == "__main__":
    main()
