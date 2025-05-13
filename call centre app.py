import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from supabase import create_client, Client

def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        if not url.startswith("https://"):
            url = f"https://{url}"
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        if "Name or service not known" in str(e):
            st.error("DNS resolution failed. Check that your Supabase URL is correct and that you have internet connectivity.")
        elif "secrets" in str(e):
            st.error("Could not access Supabase credentials in st.secrets. Make sure you've set up your .streamlit/secrets.toml file correctly.")
        raise e

def check_db(supabase):
    try:
        user_response = supabase.table("users").select("count").limit(1).execute()
        kpi_response = supabase.table("kpis").select("count").limit(1).execute()
        perf_response = supabase.table("performance").select("count").limit(1).execute()
        st.sidebar.success("✅ Connected to database successfully")
    except Exception as e:
        st.sidebar.error(f"Database check error: {str(e)}")
        st.sidebar.info("If tables don't exist, please run the SQL setup script in Supabase. Ensure RLS policies allow access.")

def get_db_connection():
    return init_supabase()

def save_kpis(supabase, kpis):
    try:
        for metric, threshold in kpis.items():
            response = supabase.table("kpis").select("*").eq("metric", metric).execute()
            if len(response.data) == 0:
                supabase.table("kpis").insert({"metric": metric, "threshold": threshold}).execute()
            else:
                supabase.table("kpis").update({"threshold": threshold}).eq("metric", metric).execute()
        return True
    except Exception as e:
        st.error(f"Error saving KPIs: {str(e)}")
        if "violates row-level security policy" in str(e):
            st.error("You don't have permission to modify KPIs. Check your role or RLS policies.")
        return False

def get_kpis(supabase):
    try:
        response = supabase.table("kpis").select("*").execute()
        kpis = {}
        for row in response.data:
            metric = row["metric"]
            value = row["threshold"]
            if metric == "call_volume":
                kpis[metric] = int(float(value)) if value is not None else 50
            else:
                kpis[metric] = float(value) if value is not None else 0.0
        return kpis
    except Exception as e:
        st.error(f"Error retrieving KPIs: {str(e)}")
        return {}

def save_performance(supabase, agent_name, data):
    try:
        date = datetime.now().strftime("%Y-%m-%d")
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
        response = supabase.table("performance").insert(performance_data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving performance data: {str(e)}")
        if "violates row-level security policy" in str(e):
            st.error("You don't have permission to add performance data. Check your role or RLS policies.")
        return False

def get_performance(supabase, agent_name=None):
    try:
        if agent_name:
            response = supabase.table("performance").select("*").eq("agent_name", agent_name).execute()
        else:
            response = supabase.table("performance").select("*").execute()
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
        else:
            st.warning(f"No performance data found for {'agent ' + agent_name if agent_name else 'any agents'}.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving performance data: {str(e)}")
        if "violates row-level security policy" in str(e):
            st.error("RLS policy is preventing data access. Ensure you have a policy allowing agents to view their own performance data.")
        return pd.DataFrame()

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
    pass_columns = [f'{m}_pass' for m in metrics if f'{m}_pass' in results.columns]
    if pass_columns:
        results['overall_score'] = results[pass_columns].mean(axis=1) * 100
    return results

def authenticate_user(supabase, name, password):
    try:
        user_response = supabase.table("users").select("*").eq("name", name).execute()
        if user_response.data:
            return True, name, user_response.data[0]["role"]
        else:
            return False, None, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None, None

def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f0f2f5;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            color: #333333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 5px 15px;
        }
        .stButton>button:hover {
            background-color: #45a049;
            color: white;
        }
        .stHeader {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    try:
        supabase = init_supabase()
        check_db(supabase)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        st.stop()

    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None

    if not st.session_state.user:
        st.title("🔐 Login")
        with st.form("login_form"):
            name = st.text_input("Name")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                success, user, role = authenticate_user(supabase, name, password)
                if success:
                    st.session_state.user = user
                    st.session_state.role = role
                    st.success(f"Logged in as {user} ({role})")
                    st.rerun()
                else:
                    st.error("Login failed. Invalid credentials or user not found.")
        st.info("Note: For production, you should use Supabase Authentication which provides secure user management.")
        return

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()
    
    with st.sidebar.expander("ℹ️ Help & Instructions"):
        st.write("""
        - **Login**: Use your name and password to log in.
        - **Managers**: Set KPIs, input performance data, and view assessments.
        - **Agents**: View your performance metrics, history, goals, and submit feedback.
        - **Date Filter**: Use the date pickers to filter data.
        - **Trends**: Add performance data with multiple dates to see trends.
        """)
    st.sidebar.info(f"👤 Logged in as: {st.session_state.user}")
    st.sidebar.info(f"🎓 Role: {st.session_state.role}")

    if st.session_state.role == "Manager":
        st.title("📊 Manager Dashboard")
        if st.button("🔄 Refresh Data"):
            st.rerun()
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
        tabs = st.tabs(["Set KPIs", "Input Performance", "View Assessments"])

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
                    if save_kpis(supabase, new_kpis):
                        st.success("KPIs saved successfully!")
                    else:
                        st.error("Failed to save KPIs. Check your permissions.")

        with tabs[1]:
            st.header("📝 Input Agent Performance")
            try:
                response = supabase.table("users").select("*").eq("role", "Agent").execute()
                agents = [user["name"] for user in response.data]
                if not agents:
                    st.warning("No agents found in the system. Please add agents in the Supabase dashboard.")
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
                            if save_performance(supabase, agent, data):
                                st.success(f"Performance data saved for {agent}!")
                            else:
                                st.error("Failed to save performance data. Check your permissions.")
            except Exception as e:
                st.error(f"Error loading agents: {str(e)}")

        with tabs[2]:
            st.header("📊 Assessment Results")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=pd.to_datetime('2025-05-01'))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now().date())
            
            performance_df = get_performance(supabase)
            if not performance_df.empty:
                performance_df['date'] = pd.to_datetime(performance_df['date'])
                masked_df = performance_df[(performance_df['date'] >= pd.to_datetime(start_date)) & 
                                        (performance_df['date'] <= pd.to_datetime(end_date))]
                kpis = get_kpis(supabase)
                results = assess_performance(masked_df, kpis)
                st.dataframe(results)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=results.to_csv(index=False),
                    file_name="performance_data.csv",
                    mime="text/csv"
                )
                st.subheader("📈 Performance Overview")
                try:
                    fig = px.bar(results, x='agent_name', y='overall_score', color='agent_name', 
                                title="Agent Overall Scores", labels={'overall_score': 'Score (%)'},
                                hover_data=['quality_score', 'csat', 'attendance'])
                    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig)
                    
                    if 'date' in results.columns:
                        st.subheader("📉 Performance Trends")
                        dates = sorted(results['date'].unique())
                        if len(dates) > 1:
                            fig2 = px.line(results, x='date', y='overall_score', color='agent_name',
                                          title="Score Trends Over Time", labels={'overall_score': 'Score (%)'})
                            st.plotly_chart(fig2)
                        else:
                            st.info("Performance trends require data from multiple dates. Please add more performance records with different dates.")
                except Exception as e:
                    st.error(f"Error plotting data: {str(e)}")

    elif st.session_state.role == "Agent":
        st.title(f"👤 Agent Dashboard - {st.session_state.user}")
        if st.session_state.user == "Joseph Kavuma":
            try:
                st.image("Joseph.jpg", caption="Agent Profile", width=150)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime('2025-05-01'))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now().date())
        
        performance_df = get_performance(supabase, st.session_state.user)
        all_performance_df = get_performance(supabase)
        if not performance_df.empty and not all_performance_df.empty:
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
            avg_metrics = results[[
                'overall_score', 'quality_score', 'csat', 'attendance', 
                'resolution_rate', 'contact_success_rate', 'aht', 
                'talk_time'
            ]].mean()
            total_call_volume = results['call_volume'].sum()
            
            if avg_overall_score < kpis.get('overall_score', 70.0):
                st.warning("⚠️ Your average performance score is below the target. Please improve!")
            
            st.subheader("📋 Your Performance Metrics (Averages)")
            col1, col2, col3 = st.columns(3, gap="medium")
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
            
            st.subheader("📋 Your Performance History")
            st.dataframe(results)
            
            try:
                st.subheader("📈 Your Score Over Time (Monthly)")
                results['date'] = pd.to_datetime(results['date'])
                results['year_month'] = results['date'].dt.to_period('M').astype(str)
                monthly_scores = results.groupby('year_month')['overall_score'].mean().reset_index()
                fig = px.line(
                    monthly_scores, 
                    x='year_month', 
                    y='overall_score', 
                    title="Your Monthly Score Trend",
                    labels={'overall_score': 'Score (%)', 'year_month': 'Month'}
                )
                st.plotly_chart(fig)
                
                st.subheader("📊 Performance by Category (Averages)")
                metrics_df = results[[
                    'quality_score', 'attendance', 'resolution_rate', 
                    'product_knowledge', 'contact_success_rate'
                ]].mean().reset_index()
                metrics_df.columns = ['Metric', 'Average']
                fig2 = px.bar(
                    metrics_df, 
                    x='Metric', 
                    y='Average', 
                    title="Your Average Metrics",
                    labels={'Average': 'Score (%)'}
                )
                st.plotly_chart(fig2)
                
                st.subheader("📊 Comparison with Peers")
                peer_avg = all_results.groupby('agent_name')['overall_score'].mean().reset_index()
                peer_avg = peer_avg[peer_avg['agent_name'] != st.session_state.user]
                fig3 = px.box(peer_avg, y='overall_score', title="Peer Overall Score Distribution",
                             labels={'overall_score': 'Score (%)'}, points="all")
                fig3.add_hline(y=avg_overall_score, line_dash="dash", line_color="red",
                              annotation_text=f"Your Score: {avg_overall_score:.1f}%")
                st.plotly_chart(fig3)
                
                st.subheader("🎯 Your Goals")
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
                            if metric == 'aht':
                                progress = min((kpis.get(metric, 600) - current_value) / (kpis.get(metric, 600) - row['target_value']) * 100, 100) if kpis.get(metric, 600) > row['target_value'] else 0
                            else:
                                progress = min(current_value / row['target_value'] * 100, 100) if row['target_value'] > 0 else 0
                            st.progress(int(progress))
                            st.write(f"{metric.replace('_', ' ').title()}: Target {row['target_value']:.1f}{' sec' if metric == 'aht' else '%'}, Current {current_value:.1f}{' sec' if metric == 'aht' else '%'}, Status: {row['status']}")
                            if st.button(f"Update {metric} Goal", key=f"update_{metric}"):
                                new_target = st.number_input(f"New Target for {metric}", value=float(row['target_value']))
                                supabase.table("goals").update({"target_value": new_target}).eq("id", row['id']).execute()
                                st.success("Goal updated! (Pending manager approval)")
                        else:
                            st.write(f"No goal set for {metric.replace('_', ' ').title()}. Contact your manager to set it.")
                else:
                    st.info("No goals set yet. Contact your manager to set goals for all metrics.")
                
                st.subheader("💬 Submit Feedback")
                with st.form("feedback_form"):
                    feedback_text = st.text_area("Your Feedback or Suggestion")
                    if st.form_submit_button("Submit Feedback"):
                        supabase.table("feedback").insert({
                            "agent_name": st.session_state.user,
                            "message": feedback_text
                        }).execute()
                        st.success("Feedback submitted! A manager will review it soon.")
            except Exception as e:
                st.error(f"Error plotting data: {str(e)}")
                st.write("Raw data:")
                st.write(results)
        else:
            st.info("No performance data available for you yet. Please contact your manager to ensure your performance data has been entered.")
            st.write("Debug: Check the following:")
            st.write(f"- Ensure performance data exists for {st.session_state.user} in the 'performance' table.")
            st.write("- Verify RLS policies allow you to view your own data.")
            st.write("- Confirm that the 'agent_name' in the performance table matches your name exactly.")

if __name__ == "__main__":
    main()
