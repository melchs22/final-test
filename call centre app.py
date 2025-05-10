import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from httpx_oauth.clients.google import GoogleOAuth2
from datetime import datetime, timedelta
import json

# Initialize Supabase client
def init_supabase():
    try:
        url = st.secrets["supabase_url"]
        key = st.secrets["supabase_key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase: {e}")
        return None

# Initialize Google OAuth client
def init_google_oauth():
    try:
        client_id = st.secrets["google_client_id"]
        client_secret = st.secrets["google_client_secret"]
        return GoogleOAuth2(client_id=client_id, client_secret=client_secret)
    except Exception as e:
        st.error(f"Failed to initialize Google OAuth: {e}")
        return None

# Authentication
def authenticate():
    if "user_email" not in st.session_state or "user_role" not in st.session_state:
        st.write("## Welcome to Call Center Assessment System")
        
        # This would typically redirect to Google OAuth flow
        # For this demo, we'll use a simplified login
        with st.form("login_form"):
            email = st.text_input("Email")
            login_button = st.form_submit_button("Login with Google")
            
            if login_button and email:
                # Check if user exists in Supabase
                supabase = init_supabase()
                if supabase:
                    response = supabase.table("users").select("email", "role").eq("email", email).execute()
                    if response.data:
                        user = response.data[0]
                        st.session_state["user_email"] = user["email"]
                        st.session_state["user_role"] = user["role"]
                        st.experimental_rerun()
                    else:
                        st.error("User not found. Please contact your administrator.")
        
        # Debug info
        st.write("DEBUG: Login screen shown. No user in session.")
        return False
    else:
        # Debug info
        st.write(f"DEBUG: Logged-in user: {st.session_state.user_email}, Role: {st.session_state.user_role}")
        return True

# Clear cache button for debugging
def clear_cache():
    st.cache_data.clear()
    st.experimental_rerun()

# Manager Dashboard
def manager_dashboard(supabase):
    st.title("Manager Dashboard")
    
    # Debug button
    if st.button("Clear Cache (Debug)"):
        clear_cache()
    
    tabs = st.tabs(["Set KPIs", "Input Performance", "View Assessments", "Manage Users"])
    
    # Set KPIs tab
    with tabs[0]:
        st.header("Set Key Performance Indicators")
        
        # Get existing KPIs
        kpis_response = supabase.table("kpis").select("*").execute()
        if kpis_response.data:
            st.write("Current KPI Thresholds:")
            kpis_df = pd.DataFrame(kpis_response.data)
            st.dataframe(kpis_df)
        
        # Form to update KPIs
        with st.form("kpi_form"):
            st.subheader("Update KPI Thresholds")
            metric = st.selectbox("Metric", ["attendance", "quality_score", "product_knowledge", 
                                            "contact_success_rate", "onboarding", "reporting",
                                            "talk_time", "resolution_rate", "aht", "csat", "call_volume"])
            threshold = st.number_input("Threshold Value", min_value=0.0, step=0.1)
            submit_kpi = st.form_submit_button("Save KPI")
            
            if submit_kpi:
                try:
                    # Upsert KPI
                    response = supabase.table("kpis").upsert({"metric": metric, "threshold": threshold}).execute()
                    st.success(f"KPI threshold for {metric} updated successfully!")
                except Exception as e:
                    st.error(f"Failed to update KPI: {e}")
    
    # Input Performance tab
    with tabs[1]:
        st.header("Input Agent Performance")
        
        # Get agents
        users_response = supabase.table("users").select("email").eq("role", "Agent").execute()
        if users_response.data:
            agents = [user["email"] for user in users_response.data]
            
            with st.form("performance_form"):
                st.subheader("Add Performance Data")
                agent_email = st.selectbox("Select Agent", agents)
                perf_date = st.date_input("Date", datetime.now().date())
                
                # Performance metrics
                attendance = st.slider("Attendance (%)", 0.0, 100.0, 95.0, 0.1)
                quality_score = st.slider("Quality Score (%)", 0.0, 100.0, 80.0, 0.1)
                product_knowledge = st.slider("Product Knowledge (%)", 0.0, 100.0, 85.0, 0.1)
                contact_success_rate = st.slider("Contact Success Rate (%)", 0.0, 100.0, 75.0, 0.1)
                onboarding = st.slider("Onboarding (%)", 0.0, 100.0, 90.0, 0.1)
                reporting = st.slider("Reporting (%)", 0.0, 100.0, 85.0, 0.1)
                talk_time = st.slider("Talk Time (%)", 0.0, 100.0, 70.0, 0.1)
                resolution_rate = st.slider("Resolution Rate (%)", 0.0, 100.0, 80.0, 0.1)
                aht = st.number_input("Average Handle Time (seconds)", 100, 1000, 300, 10)
                csat = st.slider("Customer Satisfaction (%)", 0.0, 100.0, 85.0, 0.1)
                call_volume = st.number_input("Call Volume", 0, 500, 100, 5)
                
                submit_perf = st.form_submit_button("Save Performance Data")
                
                if submit_perf:
                    try:
                        # Insert performance data
                        perf_data = {
                            "agent_email": agent_email,
                            "date": perf_date.isoformat(),
                            "attendance": attendance,
                            "quality_score": quality_score,
                            "product_knowledge": product_knowledge,
                            "contact_success_rate": contact_success_rate,
                            "onboarding": onboarding,
                            "reporting": reporting,
                            "talk_time": talk_time,
                            "resolution_rate": resolution_rate,
                            "aht": aht,
                            "csat": csat,
                            "call_volume": call_volume
                        }
                        
                        response = supabase.table("performance").insert(perf_data).execute()
                        st.success(f"Performance data for {agent_email} added successfully!")
                    except Exception as e:
                        st.error(f"Failed to add performance data: {e}")
        else:
            st.warning("No agents found. Please add users first.")
    
    # View Assessments tab
    with tabs[2]:
        st.header("View Assessments")
        
        # Get performance data
        perf_response = supabase.table("performance").select("*").execute()
        kpis_response = supabase.table("kpis").select("*").execute()
        
        if perf_response.data and kpis_response.data:
            # Convert to dataframes
            perf_df = pd.DataFrame(perf_response.data)
            kpis_df = pd.DataFrame(kpis_response.data).set_index("metric")
            
            # Add pass/fail status
            for metric in kpis_df.index:
                if metric in perf_df.columns:
                    threshold = kpis_df.loc[metric, "threshold"]
                    
                    # For AHT, lower is better
                    if metric == "aht":
                        perf_df[f"{metric}_status"] = perf_df[metric].apply(
                            lambda x: "Pass" if x <= threshold else "Fail")
                    else:
                        perf_df[f"{metric}_status"] = perf_df[metric].apply(
                            lambda x: "Pass" if x >= threshold else "Fail")
            
            # Display assessment table
            st.subheader("Performance Assessment")
            st.dataframe(perf_df)
            
            # Overall performance chart
            st.subheader("Overall Agent Performance")
            
            # Calculate average scores
            avg_scores = perf_df.groupby("agent_email")[["attendance", "quality_score", "csat"]].mean().reset_index()
            
            # Create bar chart
            fig = px.bar(
                avg_scores, 
                x="agent_email", 
                y=["attendance", "quality_score", "csat"],
                barmode="group",
                title="Average Performance by Agent",
                labels={"agent_email": "Agent", "value": "Score (%)", "variable": "Metric"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No assessment data available yet.")
    
    # Manage Users tab
    with tabs[3]:
        st.header("Manage Users")
        
        # Display existing users
        users_response = supabase.table("users").select("*").execute()
        if users_response.data:
            st.subheader("Current Users")
            users_df = pd.DataFrame(users_response.data)
            st.dataframe(users_df)
        
        # Form to add users
        with st.form("user_form"):
            st.subheader("Add New User")
            user_email = st.text_input("Email")
            user_role = st.selectbox("Role", ["Agent", "Manager"])
            submit_user = st.form_submit_button("Add User")
            
            if submit_user and user_email:
                try:
                    # Insert user
                    response = supabase.table("users").insert({"email": user_email, "role": user_role}).execute()
                    st.success(f"User {user_email} added successfully as {user_role}!")
                except Exception as e:
                    st.error(f"Failed to add user: {e}")

# Agent Dashboard
def agent_dashboard(supabase, user_email):
    st.title("Agent Dashboard")
    
    # Debug button
    if st.button("Clear Cache (Debug)"):
        clear_cache()
    
    # Get performance data for this agent
    perf_response = supabase.table("performance").select("*").eq("agent_email", user_email).execute()
    kpis_response = supabase.table("kpis").select("*").execute()
    
    # Debug output
    st.write(f"DEBUG: Performance data rows: {len(perf_response.data if perf_response.data else [])}")
    
    # Section: Input Your Goals
    st.header("Input Your Goals")
    st.write("DEBUG: Rendering Goal Input Section")
    
    with st.form("goal_form"):
        st.subheader("Add a New Goal")
        goal_metric = st.selectbox("Metric", ["attendance", "quality_score", "product_knowledge", 
                                           "contact_success_rate", "onboarding", "reporting",
                                           "talk_time", "resolution_rate", "aht", "csat", "call_volume"])
        
        # Set appropriate min/max values based on metric
        if goal_metric == "aht":
            goal_value = st.number_input("Target Value (seconds)", 100, 1000, 250, 10)
        elif goal_metric == "call_volume":
            goal_value = st.number_input("Target Value (calls)", 10, 500, 120, 5)
        else:
            goal_value = st.number_input("Target Value (%)", 0.0, 100.0, 90.0, 0.1)
        
        submit_goal = st.form_submit_button("Save Goal")
        
        if submit_goal:
            try:
                # Insert goal
                goal_data = {
                    "agent_email": user_email,
                    "metric": goal_metric,
                    "goal_value": goal_value,
                    "set_date": datetime.now().date().isoformat()
                }
                
                response = supabase.table("agent_goals").insert(goal_data).execute()
                st.success(f"Goal for {goal_metric} added successfully!")
            except Exception as e:
                st.error(f"Failed to add goal: {e}")
    
    # Section: Manage Your Goals
    st.header("Manage Your Goals")
    
    # Get goals for this agent
    goals_response = supabase.table("agent_goals").select("*").eq("agent_email", user_email).execute()
    
    # Debug output
    st.write(f"DEBUG: Goals data rows: {len(goals_response.data if goals_response.data else [])}")
    
    if goals_response.data:
        goals_df = pd.DataFrame(goals_response.data)
        
        # Add current value from performance data
        if perf_response.data:
            perf_df = pd.DataFrame(perf_response.data)
            # Get the latest performance data
            latest_perf = perf_df.sort_values("date", ascending=False).drop_duplicates(subset=["agent_email", "date"]).reset_index(drop=True)
            
            # Calculate progress for each goal
            for idx, goal in goals_df.iterrows():
                metric = goal["metric"]
                goal_value = goal["goal_value"]
                
                if metric in latest_perf.columns:
                    current_value = latest_perf[metric].iloc[0]
                    goals_df.at[idx, "current_value"] = current_value
                    
                    # Calculate progress (for AHT, lower is better)
                    if metric == "aht":
                        progress = min(100, max(0, (goal_value / current_value * 100)))
                    else:
                        progress = min(100, max(0, (current_value / goal_value * 100)))
                    
                    goals_df.at[idx, "progress"] = progress
                else:
                    goals_df.at[idx, "current_value"] = None
                    goals_df.at[idx, "progress"] = 0
        
        # Display goals table
        st.subheader("Your Goals")
        
        # Display each goal with update/delete options in expanders
        for idx, goal in goals_df.iterrows():
            with st.expander(f"Goal: {goal['metric']} - Target: {goal['goal_value']}"):
                cols = st.columns([3, 1, 1])
                
                # Goal details
                cols[0].write(f"**ID:** {goal['id']}")
                cols[0].write(f"**Metric:** {goal['metric']}")
                cols[0].write(f"**Target:** {goal['goal_value']}")
                if "current_value" in goal and goal["current_value"] is not None:
                    cols[0].write(f"**Current:** {goal['current_value']}")
                if "progress" in goal:
                    cols[0].write(f"**Progress:** {goal['progress']:.1f}%")
                cols[0].write(f"**Set Date:** {goal['set_date']}")
                
                # Update form
                with cols[1].form(f"update_goal_{goal['id']}"):
                    st.write("Update Goal")
                    if goal["metric"] == "aht":
                        new_value = st.number_input("New Target", 100, 1000, int(goal["goal_value"]), 10)
                    elif goal["metric"] == "call_volume":
                        new_value = st.number_input("New Target", 10, 500, int(goal["goal_value"]), 5)
                    else:
                        new_value = st.number_input("New Target", 0.0, 100.0, float(goal["goal_value"]), 0.1)
                    
                    update_button = st.form_submit_button("Update")
                    
                    if update_button:
                        try:
                            # Update goal
                            response = supabase.table("agent_goals").update({
                                "goal_value": new_value,
                                "set_date": datetime.now().date().isoformat()
                            }).eq("id", goal["id"]).execute()
                            st.success("Goal updated!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to update goal: {e}")
                
                # Delete button
                with cols[2].form(f"delete_goal_{goal['id']}"):
                    st.write("Delete Goal")
                    delete_button = st.form_submit_button("Delete")
                    
                    if delete_button:
                        try:
                            # Delete goal
                            response = supabase.table("agent_goals").delete().eq("id", goal["id"]).execute()
                            st.success("Goal deleted!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to delete goal: {e}")
                
                # Progress gauge
                if "progress" in goal:
                    progress = goal["progress"]
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=progress,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": f"Progress toward {goal['metric']} goal"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "green" if progress >= 80 else "orange"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 80], "color": "gray"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 80
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("You haven't set any goals yet. Use the form above to add your first goal!")
    
    # Section: Metric Breakdown
    st.header("Metric Breakdown")
    
    if perf_response.data and kpis_response.data:
        # Convert to dataframes
        perf_df = pd.DataFrame(perf_response.data)
        kpis_df = pd.DataFrame(kpis_response.data).set_index("metric")
        
        # Metric selection
        available_metrics = [col for col in perf_df.columns if col not in ["id", "agent_email", "date"]]
        selected_metric = st.selectbox("Select metric to view", available_metrics)
        
        # Create line chart
        fig = px.line(
            perf_df.sort_values("date"), 
            x="date", 
            y=selected_metric,
            title=f"{selected_metric.replace('_', ' ').title()} Over Time",
            markers=True
        )
        
        # Add KPI threshold if available
        if selected_metric in kpis_df.index:
            threshold = kpis_df.loc[selected_metric, "threshold"]
            fig.add_shape(
                type="line",
                line=dict(dash="dash", color="red", width=2),
                y0=threshold,
                y1=threshold,
                x0=perf_df["date"].min(),
                x1=perf_df["date"].max()
            )
            fig.add_annotation(
                x=perf_df["date"].max(),
                y=threshold,
                text=f"KPI: {threshold}",
                showarrow=False,
                yshift=10
            )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No performance data available yet.")
    
    # Section: Additional Features
    st.header("Additional Features")
    
    features_tabs = st.tabs(["Performance Trends", "Recommendations", "Peer Comparison", 
                           "Feedback", "Training Resources", "Achievements", "Customize Dashboard"])
    
    # Performance Trends tab
    with features_tabs[0]:
        st.subheader("Performance Trends")
        
        if perf_response.data and kpis_response.data:
            perf_df = pd.DataFrame(perf_response.data)
            kpis_df = pd.DataFrame(kpis_response.data).set_index("metric")
            
            # Get trends for last 5 data points
            if len(perf_df) >= 2:
                perf_df = perf_df.sort_values("date", ascending=False).head(5)
                
                # Calculate trends
                trend_data = []
                for metric in kpis_df.index:
                    if metric in perf_df.columns:
                        # Calculate trend (first vs last value)
                        first_val = perf_df[metric].iloc[-1]
                        last_val = perf_df[metric].iloc[0]
                        
                        if metric == "aht":  # Lower is better for AHT
                            if last_val < first_val:
                                trend = "Improving"
                            elif last_val > first_val:
                                trend = "Declining"
                            else:
                                trend = "Stable"
                            
                            status = "Pass" if last_val <= kpis_df.loc[metric, "threshold"] else "Fail"
                        else:
                            if last_val > first_val:
                                trend = "Improving"
                            elif last_val < first_val:
                                trend = "Declining"
                            else:
                                trend = "Stable"
                            
                            status = "Pass" if last_val >= kpis_df.loc[metric, "threshold"] else "Fail"
                        
                        trend_data.append({
                            "Metric": metric,
                            "Current": last_val,
                            "Trend": trend,
                            "Status": status
                        })
                
                trend_df = pd.DataFrame(trend_data)
                
                # Display with colored status
                def highlight_status(val):
                    if val == "Pass":
                        return "background-color: lightgreen"
                    elif val == "Fail":
                        return "background-color: lightcoral"
                    return ""
                
                st.dataframe(trend_df.style.applymap(highlight_status, subset=["Status"]))
            else:
                st.info("Not enough data points to calculate trends.")
        else:
            st.info("No performance data available yet.")
    
    # Recommendations tab
    with features_tabs[1]:
        st.subheader("Personalized Recommendations")
        
        if perf_response.data and kpis_response.data:
            perf_df = pd.DataFrame(perf_response.data)
            kpis_df = pd.DataFrame(kpis_response.data).set_index("metric")
            
            # Get latest performance
            latest_perf = perf_df.sort_values("date", ascending=False).iloc[0]
            
            # Generate recommendations for failing metrics
            recommendations = []
            for metric in kpis_df.index:
                if metric in latest_perf.index:
                    threshold = kpis_df.loc[metric, "threshold"]
                    value = latest_perf[metric]
                    
                    # Check if failing KPI
                    if (metric == "aht" and value > threshold) or (metric != "aht" and value < threshold):
                        if metric == "attendance":
                            recommendations.append("Improve attendance by planning your schedule in advance.")
                        elif metric == "quality_score":
                            recommendations.append("Enhance quality by following the call script more closely.")
                        elif metric == "product_knowledge":
                            recommendations.append("Review product documentation to improve your knowledge.")
                        elif metric == "contact_success_rate":
                            recommendations.append("Focus on building rapport with customers in the first minute.")
                        elif metric == "aht":
                            recommendations.append("Reduce handle time by using CRM shortcuts and templates.")
                        elif metric == "csat":
                            recommendations.append("Improve customer satisfaction by confirming their needs are met before ending calls.")
                        else:
                            recommendations.append(f"Work on improving your {metric.replace('_', ' ')}.")
            
            if recommendations:
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.success("Great job! You're meeting all KPI targets.")
        else:
            st.info("No performance data available yet.")
    
    # Peer Comparison tab
    with features_tabs[2]:
        st.subheader("Peer Comparison")
        
        if perf_response.data:
            all_perf_response = supabase.table("performance").select("*").execute()
            all_perf_df = pd.DataFrame(all_perf_response.data)
            
            # Get latest data for all agents
            latest_dates = all_perf_df.groupby("agent_email")["date"].max().reset_index()
            latest_data = []
            for _, row in latest_dates.iterrows():
                agent_data = all_perf_df[(all_perf_df["agent_email"] == row["agent_email"]) & 
                                         (all_perf_df["date"] == row["date"])]
                latest_data.append(agent_data.iloc[0])
            
            latest_df = pd.DataFrame(latest_data)
            
            # Calculate team averages
            team_avg = latest_df.mean(numeric_only=True)
            
            # Get user's latest data
            user_data = latest_df[latest_df["agent_email"] == user_email].iloc[0]
            
            # Create comparison dataframe
            comparison_data = []
            for metric in ["attendance", "quality_score", "csat"]:
                comparison_data.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Your Score": user_data[metric],
                    "Team Average": team_avg[metric]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison chart
            fig = px.bar(
                comparison_df,
                x="Metric",
                y=["Your Score", "Team Average"],
                barmode="group",
                title="Your Performance vs. Team Average",
                labels={"value": "Score (%)", "variable": ""}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet.")
    
    # Feedback tab
    with features_tabs[3]:
        st.subheader("Submit Feedback")
        
        with st.form("feedback_form"):
            feedback_text = st.text_area("Share your thoughts or suggestions", height=150)
            submit_feedback = st.form_submit_button("Submit Feedback")
            
            if submit_feedback and feedback_text:
                try:
                    # Insert feedback
                    feedback_data = {
                        "agent_email": user_email,
                        "feedback": feedback_text,
                        "submitted_date": datetime.now().date().isoformat()
                    }
                    
                    response = supabase.table("agent_feedback").insert(feedback_data).execute()
                    st.success("Feedback submitted successfully!")
                except Exception as e:
                    st.error(f"Failed to submit feedback: {e}")
    
    # Training Resources tab
    with features_tabs[4]:
        st.subheader("Training Resources")
        
        # Get training resources
        resources_response = supabase.table("training_resources").select("*").execute()
        
        if resources_response.data:
            # Display resources
            resources_df = pd.DataFrame(resources_response.data)
            
            # Group by metric
            for metric in resources_df["metric"].unique():
                st.write(f"**{metric.replace('_', ' ').title()}**")
                
                metric_resources = resources_df[resources_df["metric"] == metric]
                for _, resource in metric_resources.iterrows():
                    st.write(f"• [{resource['resource_name']}]({resource['resource_url']})")
        else:
            st.info("No training resources available yet.")
    
    # Achievements tab
    with features_tabs[5]:
        st.subheader("Your Achievements")
        
        # Get user achievements
        achievements_response = supabase.table("agent_achievements").select("*").eq("agent_email", user_email).execute()
        
        if achievements_response.data:
            achievements_df = pd.DataFrame(achievements_response.data)
            
            # Display achievements
            for _, achievement in achievements_df.iterrows():
                st.write(f"🏆 **{achievement['achievement_name']}** - Earned on {achievement['date_earned']}")
        else:
            # Check if we can award any achievements based on performance
            if perf_response.data:
                perf_df = pd.DataFrame(perf_response.data)
                
                # Check for perfect attendance
                if "attendance" in perf_df.columns and (perf_df["attendance"] >= 100).any():
                    st.write("🏆 **Perfect Attendance** - Achieved 100% attendance")
                
                # Check for high CSAT
                if "csat" in perf_df.columns and (perf_df["csat"] >= 95).any():
                    st.write("🏆 **Customer Champion** - Achieved 95%+ CSAT")
                
                # Placeholder for more achievements
                st.info("Keep up the good work to earn more achievements!")
            else:
                st.info("No achievements yet. Meet your performance goals to earn achievements!")
    
    # Customize Dashboard tab
    with features_tabs[6]:
        st.subheader("Customize Your Dashboard")
        
        # Get user preferences
        prefs_response = supabase.table("agent_preferences").select("*").eq("agent_email", user_email).execute()
        
        current_prefs = []
        if prefs_response.data:
            current_prefs = prefs_response.data[0].get("preferred_metrics", [])
        
        # Form to update preferences
        with st.form("preferences_form"):
            metrics = ["attendance", "quality_score", "product_knowledge", "contact_success_rate", 
                      "onboarding", "reporting", "talk_time", "resolution_rate", "aht", "csat", "call_volume"]
            
            selected_metrics = st.multiselect("Select metrics to display on your dashboard", metrics, default=current_prefs)
            submit_prefs = st.form_submit_button("Save Preferences")
            
            if submit_prefs:
                try:
                    # Upsert preferences
                    response = supabase.table("agent_preferences").upsert({
                        "agent_email": user_email,
                        "preferred_metrics": selected_metrics
                    }).execute()
                    st.success("Dashboard preferences saved!")
                except Exception as e:
                    st.error(f"Failed to save preferences: {e}")
        
        # Display custom metrics if preferences saved
        if prefs_response.data and perf_response.data:
            st.subheader("Your Custom Dashboard")
            
            perf_df = pd.DataFrame(perf_response.data).sort_values("date")
            preferred_metrics = prefs_response.data[0].get("preferred_metrics", [])
            
            if preferred_metrics:
                for metric in preferred_metrics:
                    if metric in perf_df.columns:
                        fig = px.line(
                            perf_df, 
                            x="date", 
                            y=metric,
                            title=f"{metric.replace('_', ' ').title()} Over Time",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select metrics above to customize your dashboard.")

# Main application
def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    
    # Initialize Supabase client
    supabase = init_supabase()
    if not supabase:
        st.error("Failed to connect to database. Please check your connection and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("Call Center Assessment")
        
        # Logout button
        if "user_email" in st.session_state:
            if st.button("Logout"):
                # Clear session state and redirect to login
                st.session_state.clear()
                st.experimental_rerun()
    
    # Authenticate user
    if authenticate():
        # Display appropriate dashboard based on role
        if st.session_state.user_role == "Manager":
            manager_dashboard(supabase)
        else:
            agent_dashboard(supabase, st.session_state.user_email)

if __name__ == "__main__":
    main()
