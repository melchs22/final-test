import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from supabase import create_client, Client
import httpx
from httpx_oauth.clients.google import GoogleOAuth2
import os
import asyncio
import plotly.graph_objects as go

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
    res = supabase.table("users").select("role").eq("email", email).execute()
    return res.data[0]["role"] if res.data else None

def get_agents():
    res = supabase.table("users").select("email").eq("role", "Agent").execute()
    return [user["email"] for user in res.data]

# Assessment
def assess_performance(performance_df, kpis):
    results = performance_df.copy()
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']
    for metric in metrics:
        if metric == 'aht':
            results[f'{metric}_pass'] = results[metric] <= kpis.get(metric, 600)
        else:
            results[f'{metric}_pass'] = results[metric] >= kpis.get(metric, 50)
    results['overall_score'] = results[[f'{m}_pass' for m in metrics]].mean(axis=1) * 100
    return results

# Goal Functions
def save_goal(agent_email, metric, goal_value):
    supabase.table("agent_goals").insert({
        "agent_email": agent_email,
        "metric": metric,
        "goal_value": goal_value,
        "set_date": datetime.now().strftime("%Y-%m-%d")
    }).execute()

def update_goal(goal_id, goal_value):
    supabase.table("agent_goals").update({
        "goal_value": goal_value,
        "set_date": datetime.now().strftime("%Y-%m-%d")
    }).eq("id", goal_id).execute()

def delete_goal(goal_id):
    supabase.table("agent_goals").delete().eq("id", goal_id).execute()

def get_goals(agent_email):
    res = supabase.table("agent_goals").select("*").eq("agent_email", agent_email).execute()
    return pd.DataFrame(res.data)

# Feedback Functions
def save_feedback(agent_email, feedback):
    supabase.table("agent_feedback").insert({
        "agent_email": agent_email,
        "feedback": feedback,
        "submitted_date": datetime.now().strftime("%Y-%m-%d")
    }).execute()

# Training Resources Functions
def get_resources(metric):
    res = supabase.table("training_resources").select("*").eq("metric", metric).execute()
    return res.data

# Achievement Functions
def award_achievement(agent_email, achievement_name):
    existing = supabase.table("agent_achievements").select("*").eq("agent_email", agent_email).eq("achievement_name", achievement_name).execute()
    if not existing.data:
        supabase.table("agent_achievements").insert({
            "agent_email": agent_email,
            "achievement_name": achievement_name,
            "date_earned": datetime.now().strftime("%Y-%m-%d")
        }).execute()

def get_achievements(agent_email):
    res = supabase.table("agent_achievements").select("*").eq("agent_email", agent_email).execute()
    return res.data

# Preferences Functions
def save_preferences(agent_email, metrics):
    supabase.table("agent_preferences").upsert({
        "agent_email": agent_email,
        "preferred_metrics": metrics
    }).execute()

def get_preferences(agent_email):
    res = supabase.table("agent_preferences").select("preferred_metrics").eq("agent_email", agent_email).execute()
    return res.data[0]['preferred_metrics'] if res.data else ['attendance', 'quality_score', 'aht', 'csat']

# User Management Functions
def add_user(email, role):
    supabase.table("users").insert({"email": email, "role": role}).execute()

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
                        st.error("User not registered in the system. Please contact the administrator to register your email.")
                except Exception as e:
                    st.error(f"Login failed: {str(e)}. Please try again or contact support.")
        return

    if st.button("Logout"):
        for key in ["user", "role", "oauth_client", "oauth_token"]:
            st.session_state.pop(key, None)
        st.rerun()

    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']

    if st.session_state.role == "Manager":
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

        with tabs[3]:
            st.header("Manage Users")
            with st.form("user_form"):
                new_email = st.text_input("User Email")
                new_role = st.selectbox("Role", ["Agent", "Manager"])
                if st.form_submit_button("Add User"):
                    try:
                        add_user(new_email, new_role)
                        st.success(f"User {new_email} added as {new_role}!")
                    except Exception as e:
                        st.error(f"Failed to add user: {str(e)}")

    elif st.session_state.role == "Agent":
        st.title(f"Agent Dashboard - {st.session_state.user}")
        performance_df = get_performance(st.session_state.user)
        kpis = get_kpis()
        st.write(f"DEBUG: Performance data rows: {len(performance_df)}")
        
        if not performance_df.empty:
            results = assess_performance(performance_df, kpis)
            st.dataframe(results)
            fig = px.line(results, x='date', y='overall_score', title="Your Score Over Time")
            st.plotly_chart(fig)
        else:
            st.write("No performance data available.")

        # Enhanced Goal Setting and Progress Tracking
        st.subheader("Set and Manage Your Goals")
        st.write("DEBUG: Rendering Goal Setting")
        with st.form("goal_form"):
            st.write("Set New Goal")
            metric = st.selectbox("Select Metric", metrics, key="new_goal_metric")
            goal_value = st.number_input("Target Value", min_value=0.0, max_value=100.0 if metric != 'aht' else 1000.0, step=0.1)
            if st.form_submit_button("Add Goal"):
                if metric == 'aht' and goal_value < 100:
                    st.error("AHT target should be realistic (e.g., 100-1000 seconds).")
                elif metric != 'aht' and goal_value > 100:
                    st.error("Percentage metrics should be between 0 and 100.")
                else:
                    save_goal(st.session_state.user, metric, goal_value)
                    st.success(f"Goal for {metric} set to {goal_value}!")

        goals_df = get_goals(st.session_state.user)
        st.write(f"DEBUG: Goals data rows: {len(goals_df)}")
        if not goals_df.empty:
            st.write("Your Current Goals")
            for _, goal in goals_df.iterrows():
                goal_id = goal['id']
                metric = goal['metric']
                goal_value = goal['goal_value']
                set_date = goal['set_date']
                latest_value = performance_df[metric].iloc[-1] if not performance_df.empty and metric in performance_df else 0
                progress = (latest_value / goal_value * 100) if metric != 'aht' else (goal_value / latest_value * 100 if latest_value > 0 else 0)
                progress = min(progress, 100)

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{metric.replace('_', ' ').title()}**: Target = {goal_value}, Current = {latest_value:.2f}, Set on {set_date}")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=progress,
                        title={'text': f"Progress to {metric.replace('_', ' ').title()} Goal"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green" if progress >= 80 else "orange"}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    new_value = st.number_input(f"Update {metric} Target", min_value=0.0, max_value=100.0 if metric != 'aht' else 1000.0, value=goal_value, key=f"update_{goal_id}")
                    if st.button("Update", key=f"update_btn_{goal_id}"):
                        update_goal(goal_id, new_value)
                        st.success(f"Updated {metric} goal to {new_value}!")
                with col3:
                    if st.button("Delete", key=f"delete_btn_{goal_id}"):
                        delete_goal(goal_id)
                        st.success(f"Deleted {metric} goal!")
        else:
            st.write("No goals set yet.")

        # 1. Detailed Metric Breakdown
        st.subheader("Metric Breakdown")
        st.write("DEBUG: Rendering Metric Breakdown")
        selected_metric = st.selectbox("Select Metric to View", metrics)
        if not performance_df.empty:
            fig = px.line(performance_df, x='date', y=selected_metric, title=f"Your {selected_metric.replace('_', ' ').title()} Over Time")
            kpi_threshold = kpis.get(selected_metric, 50)
            fig.add_hline(y=kpi_threshold, line_dash="dash", line_color="red", annotation_text=f"KPI: {kpi_threshold}")
            st.plotly_chart(fig)

        # 2. Performance Trends and Alerts
        st.subheader("Performance Trends")
        st.write("DEBUG: Rendering Performance Trends")
        if not performance_df.empty:
            trends = {}
            for metric in metrics:
                last_five = performance_df[metric].tail(5)
                if len(last_five) >= 2:
                    trend = "Improving" if last_five.diff().mean() > 0 else "Declining" if last_five.diff().mean() < 0 else "Stable"
                    pass_status = "Pass" if (metric == 'aht' and last_five.iloc[-1] <= kpis.get(metric, 600)) or \
                                           (metric != 'aht' and last_five.iloc[-1] >= kpis.get(metric, 50)) else "Fail"
                    trends[metric] = {"Trend": trend, "Status": pass_status}
            trends_df = pd.DataFrame(trends).T
            st.dataframe(trends_df.style.applymap(lambda x: 'color: red' if x == 'Fail' else 'color: green', subset=['Status']))

        # 3. Personalized Recommendations
        st.subheader("Personalized Recommendations")
        st.write("DEBUG: Rendering Recommendations")
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
            for metric in metrics:
                if metric == 'aht' and latest_performance[metric] > kpis.get(metric, 600):
                    st.write(f"- {recommendations.get(metric, 'Work on improving this metric.')}")
                elif metric != 'aht' and latest_performance[metric] < kpis.get(metric, 50):
                    st.write(f"- {recommendations.get(metric, 'Work on improving this metric.')}")
        else:
            st.write("No recommendations available yet.")

        # 4. Peer Comparison
        st.subheader("Compare to Team")
        st.write("DEBUG: Rendering Peer Comparison")
        all_performance = get_performance()
        if not all_performance.empty:
            team_avg = all_performance.groupby('date')[metrics].mean().reset_index().iloc[-1]
            agent_latest = performance_df.iloc[-1] if not performance_df.empty else pd.Series(0, index=metrics)
            comparison_df = pd.DataFrame({
                'Your Score': agent_latest[metrics],
                'Team Average': team_avg[metrics]
            })
            fig = px.bar(comparison_df, barmode='group', title="Your Performance vs. Team Average")
            st.plotly_chart(fig)

        # 5. Feedback Submission
        st.subheader("Submit Feedback")
        st.write("DEBUG: Rendering Feedback")
        with st.form("feedback_form"):
            feedback = st.text_area("Your Feedback")
            if st.form_submit_button("Submit"):
                save_feedback(st.session_state.user, feedback)
                st.success("Feedback submitted!")

        # 6. Historical Data Download
        st.subheader("Download Your Performance Data")
        st.write("DEBUG: Rendering Data Download")
        if not performance_df.empty:
            csv = performance_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"performance_{st.session_state.user}.csv",
                mime="text/csv"
            )

        # 7. Training Resources
        st.subheader("Training Resources")
        st.write("DEBUG: Rendering Training Resources")
        if not performance_df.empty:
            latest_performance = performance_df.iloc[-1]
            for metric in metrics:
                if metric == 'aht' and latest_performance[metric] > kpis.get(metric, 600):
                    resources = get_resources(metric)
                    for res in resources:
                        st.markdown(f"- [{res['resource_name']}]({res['resource_url']}) for {metric.replace('_', ' ').title()}")
                elif metric != 'aht' and latest_performance[metric] < kpis.get(metric, 50):
                    resources = get_resources(metric)
                    for res in resources:
                        st.markdown(f"- [{res['resource_name']}]({res['resource_url']}) for {metric.replace('_', ' ').title()}")

        # 8. Gamification Elements
        st.subheader("Your Achievements")
        st.write("DEBUG: Rendering Achievements")
        if not performance_df.empty:
            latest_performance = performance_df.iloc[-1]
            if latest_performance['attendance'] >= 100:
                award_achievement(st.session_state.user, "Perfect Attendance")
            if latest_performance['csat'] >= 95:
                award_achievement(st.session_state.user, "CSAT Star")
            if latest_performance['resolution_rate'] >= 90:
                award_achievement(st.session_state.user, "Resolution Master")
            achievements = get_achievements(st.session_state.user)
            for ach in achievements:
                st.write(f"- {ach['achievement_name']} (Earned on {ach['date_earned']})")
        else:
            st.write("No achievements yet.")

        # 9. Customizable Dashboard Widgets
        st.subheader("Customize Dashboard")
        st.write("DEBUG: Rendering Customize Dashboard")
        preferred_metrics = st.multiselect("Select Metrics to Display", metrics, default=get_preferences(st.session_state.user))
        if st.button("Save Preferences"):
            save_preferences(st.session_state.user, preferred_metrics)
            st.success("Preferences saved!")

        if not performance_df.empty:
            for metric in preferred_metrics:
                fig = px.line(performance_df, x='date', y=metric, title=f"Your {metric.replace('_', ' ').title()} Over Time")
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
