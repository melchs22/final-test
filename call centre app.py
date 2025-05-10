# Supabase-based Streamlit App
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from supabase import create_client, Client
import httpx
from httpx_oauth.clients.google import GoogleOAuth2
import os
import asyncio

# Initialize Supabase client
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

supabase = get_supabase()

# OAuth setup
def get_google_client():
    return GoogleOAuth2(
        client_id=st.secrets["oauth"]["client_id"],
        client_secret=st.secrets["oauth"]["client_secret"]
    )

async def get_authorization_url(client):
    redirect_uri = st.secrets["oauth"]["redirect_uri"]
    if "STREAMLIT_CLOUD_URL" in os.environ:
        redirect_uri = f"{os.environ['STREAMLIT_CLOUD_URL']}"
    return await client.get_authorization_url(
        redirect_uri=redirect_uri,
        scope=["email", "profile"]
    )

async def get_google_user(token):
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        response = await client.get("https://www.googleapis.com/oauth2/v3/userinfo", headers=headers)
        response.raise_for_status()
        return response.json()['email']

# KPI Functions
def save_kpis(kpis):
    for metric, threshold in kpis.items():
        supabase.table("kpis").upsert({"metric": metric, "threshold": threshold}).execute()

def get_kpis():
    res = supabase.table("kpis").select("*").execute()
    return {item["metric"]: item["threshold"] for item in res.data}

# Performance Functions
def save_performance(agent_email, data):
    data["agent_email"] = agent_email
    data["date"] = datetime.now().strftime("%Y-%m-%d")
    supabase.table("performance").insert(data).execute()

def get_performance(agent_email=None):
    if agent_email:
        response = supabase.table("performance").select("*").eq("agent_email", agent_email).execute()
    else:
        response = supabase.table("performance").select("*").execute()
    return pd.DataFrame(response.data)

# User Functions
def get_user_role(email):
    res = supabase.table("users").select("role").eq("email", email).single().execute()
    return res.data["role"] if res.data else None

def get_agents():
    res = supabase.table("users").select("email").eq("role", "Agent").execute()
    return [user["email"] for user in res.data]

# Assessment

def assess_performance(performance_df, kpis):
    results = performance_df.copy()
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'csat', 'call_volume']
    for metric in metrics:
        if metric == 'aht':
            results[f'{metric}_pass'] = results[metric] <= kpis.get(metric, 600)
        else:
            results[f'{metric}_pass'] = results[metric] >= kpis.get(metric, 50)
    results['overall_score'] = results[[f'{m}_pass' for m in metrics]].mean(axis=1) * 100
    return results

# Streamlit app
def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")

    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.oauth_client = None
        st.session_state.oauth_token = None

    if not st.session_state.user:
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
                    role = get_user_role(email)
                    if role:
                        st.session_state.user = email
                        st.session_state.role = role
                        st.success(f"Logged in as {email} ({role})")
                        st.query_params.clear()
                        st.rerun()
                    else:
                        st.error("User not registered. Contact admin.")
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
        return

    if st.button("Logout"):
        for key in ["user", "role", "oauth_client", "oauth_token"]:
            st.session_state.pop(key, None)
        st.rerun()

    if st.session_state.role == "Manager":
        st.title("Manager Dashboard")
        tabs = st.tabs(["Set KPIs", "Input Performance", "View Assessments"])

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
            with st.form("performance_form"):
                agent = st.selectbox("Select Agent", agents)
                data = {
                    'attendance': st.number_input("Attendance (%)", min_value=0.0, max_value=100.0),
                    'quality_score': st.number_input("Quality Score (%)", min_value=0.0, max_value=100.0),
                    'product_knowledge': st.number_input("Product Knowledge (%)", min_value=0.0, max_value=100.0),
                    'contact_success_rate': st.number_input("Contact Success Rate (%)", min_value=0.0, max_value=100.0),
                    'onboarding': st.number_input("Onboarding (%)", min_value=0.0, max_value=100.0),
                    'reporting': st.number_input("Reporting (%)", min_value=0.0, max_value=100.0),
                    'talk_time': st.number_input("CRM Talk Time (seconds)", min_value=0.0),
                    'resolution_rate': st.number_input("Issue Resolution Rate (%)", min_value=0.0, max_value=100.0),
                    'aht': st.number_input("Average Handle Time (seconds)", min_value=0.0),
                    'csat': st.number_input("Customer Satisfaction (%)", min_value=0.0, max_value=100.0),
                    'call_volume': st.number_input("Call Volume (calls)", min_value=0)
                }
                if st.form_submit_button("Submit Performance"):
                    save_performance(agent, data)
                    st.success("Performance data saved!")

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

    elif st.session_state.role == "Agent":
        st.title(f"Agent Dashboard - {st.session_state.user}")
        performance_df = get_performance(st.session_state.user)
        if not performance_df.empty:
            kpis = get_kpis()
            results = assess_performance(performance_df, kpis)
            st.dataframe(results)
            fig = px.line(results, x='date', y='overall_score', title="Your Score Over Time")
            st.plotly_chart(fig)
        else:
            st.write("No performance data available.")

if __name__ == "__main__":
    main()
