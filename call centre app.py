import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from supabase import create_client, Client

# Inject custom CSS to match the scorecard styling
st.markdown(
    """
    <style>
    .stDataFrame {
        border: 2px solid #d32f2f;
        border-radius: 5px;
        background-color: #fff;
    }
    .stDataFrame thead th {
        background-color: #d32f2f;
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 10px;
    }
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #f5f5f5;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #ffffff;
    }
    .stDataFrame tbody tr:last-child {
        background-color: #ffcccc;
        font-weight: bold;
    }
    .stDataFrame td {
        padding: 8px;
        text-align: center;
        border: 1px solid #ddd;
    }
    .metric-header {
        background-color: #ff5722;
        color: white;
        font-weight: bold;
        padding: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Supabase initialization with better error handling
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

# Safe database check - doesn't try to create/insert data
def check_db(supabase):
    try:
        user_response = supabase.table("users").select("count").limit(1).execute()
        kpi_response = supabase.table("kpis").select("count").limit(1).execute()
        perf_response = supabase.table("performance").select("count").limit(1).execute()
        
        st.sidebar.success("✅ Connected to database successfully")
    except Exception as e:
        st.sidebar.error(f"Database check error: {str(e)}")
        st.sidebar.info("If tables don't exist, please run the SQL setup script in Supabase. Ensure RLS policies allow access.")

# Get Supabase client
def get_db_connection():
    return init_supabase()

# Save KPIs with error handling
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

# Get KPIs with error handling
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

# Save performance data with error handling
def save_performance(supabase, agent_email, data):
    try:
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
        
        response = supabase.table("performance").insert(performance_data).execute()
        return True
    except Exception as e:
        st.error(f"Error saving performance data: {str(e)}")
        if "violates row-level security policy" in str(e):
            st.error("You don't have permission to add performance data. Check your role or RLS policies.")
        return False

# Get performance data with enhanced error handling
def get_performance(supabase, agent_email=None):
    try:
        if agent_email:
            response = supabase.table("performance").select("*").eq("agent_email", agent_email).execute()
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
            st.warning(f"No performance data found for {'agent ' + agent_email if agent_email else 'any agents'}.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving performance data: {str(e)}")
        if "violates row-level security policy" in str(e):
            st.error("RLS policy is preventing data access. Ensure you have a policy allowing agents to view their own performance data.")
        return pd.DataFrame()

# Assess performance based on KPIs
def assess_performance(performance_df, kpis):
    if performance_df.empty:
        return performance_df
        
    results = performance_df.copy()
    metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate', 
               'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']
    
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

# Improved authentication with Supabase Auth
def authenticate_user(supabase, email, password):
    try:
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            user = response.user
            user_data = supabase.table("users").select("*").eq("email", email).execute()
            if user_data.data:
                role = user_data.data[0]["role"]
            else:
                role = "User"
            return True, email, role
        except Exception as auth_e:
            user_response = supabase.table("users").select("*").eq("email", email).execute()
            if user_response.data:
                return True, email, user_response.data[0]["role"]
            else:
                return False, None, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None, None

# Streamlit app
def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    
    # Initialize Supabase client
    try:
        supabase = init_supabase()
        check_db(supabase)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        st.stop()

    # Session state for authentication
    if 'user' not in st.session_state:
        st.session_state.user = None
        st.session_state.role = None

    # Auth via Supabase Auth UI
    if not st.session_state.user:
        st.title("Login")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                success, user, role = authenticate_user(supabase, email, password)
                if success:
                    st.session_state.user = user
                    st.session_state.role = role
                    st.success(f"Logged in as {user} ({role})")
                    st.rerun()
                else:
                    st.error("Login failed. Invalid credentials or user not found.")
        
        st.info("Note: For production, you should use Supabase Authentication which provides secure user management.")
        return

    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()
    
    st.sidebar.info(f"Logged in as: {st.session_state.user}")
    st.sidebar.info(f"Role: {st.session_state.role}")

    # Manager interface
    if st.session_state.role == "Manager":
        st.title("Manager Dashboard")
        tabs = st.tabs(["Set KPIs", "Input Performance", "View Assessments"])

        with tabs[0]:
            st.header("Set KPI Thresholds")
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
            st.header("Input Agent Performance")
            try:
                response = supabase.table("users").select("*").eq("role", "Agent").execute()
                agents = [user["email"] for user in response.data]
                
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
            st.header("Assessment Results")
            performance_df = get_performance(supabase)
            if not performance_df.empty:
                kpis = get_kpis(supabase)
                results = assess_performance(performance_df, kpis)
                st.dataframe(results)
                
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download All Agent Performance as CSV",
                    data=csv,
                    file_name=f"agent_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                st.subheader("Performance Overview")
                try:
                    fig = px.bar(results, x='agent_email', y='overall_score', color='agent_email', 
                                title="Agent Overall Scores", labels={'overall_score': 'Score (%)'})
                    st.plotly_chart(fig)
                    
                    if 'date' in results.columns:
                        st.subheader("Performance Trends")
                        dates = sorted(results['date'].unique())
                        if len(dates) > 1:
                            fig2 = px.line(results, x='date', y='overall_score', color='agent_email',
                                          title="Score Trends Over Time", labels={'overall_score': 'Score (%)'})
                            st.plotly_chart(fig2)
                except Exception as e:
                    st.error(f"Error plotting data: {str(e)}")
            else:
                st.info("No performance data available yet. Add performance data in the 'Input Performance' tab.")

    # Agent interface
    elif st.session_state.role == "Agent":
        st.title(f"Agent Dashboard - {st.session_state.user}")
        
        # Get individual agent performance
        performance_df = get_performance(supabase, st.session_state.user)
        # Get all agents' performance for comparison
        all_performance_df = get_performance(supabase)
        
        if not performance_df.empty:
            kpis = get_kpis(supabase)
            results = assess_performance(performance_df, kpis)
            
            # Aggregate individual performance data to monthly
            results['date'] = pd.to_datetime(results['date'])
            monthly_results = results.groupby([results['date'].dt.to_period('M'), 'agent_email']).agg({
                'overall_score': 'mean',
                'quality_score': 'mean',
                'attendance': 'mean',
                'product_knowledge': 'mean',
                'contact_success_rate': 'mean',
                'onboarding': 'mean',
                'reporting': 'mean',
                'talk_time': 'mean',
                'resolution_rate': 'mean',
                'aht': 'mean',
                'csat': 'mean',
                'call_volume': 'mean'
            }).reset_index()
            monthly_results['date'] = monthly_results['date'].dt.to_timestamp()
            
            # Aggregate all agents' performance to monthly for comparison
            if not all_performance_df.empty:
                all_performance_df['date'] = pd.to_datetime(all_performance_df['date'])
                all_monthly_results = all_performance_df.groupby(all_performance_df['date'].dt.to_period('M')).agg({
                    'overall_score': 'mean'
                }).reset_index()
                all_monthly_results['date'] = all_monthly_results['date'].dt.to_timestamp()
                all_monthly_results['agent_email'] = 'All Agents (Average)'
            
            # Display latest performance metrics in the new table format
            st.subheader("Your Performance Metrics")
            latest = results.sort_values('date', ascending=False).iloc[0]
            
            # Define the metrics and their properties for the table
            metrics_data = [
                {
                    "Key Performance Indicator": "Attendance",
                    "Weighting": "10%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['attendance']}%",
                    "Metric Score": latest['attendance'],
                    "Balanced Score": latest['attendance'] * 0.10
                },
                {
                    "Key Performance Indicator": "Quality Score",
                    "Weighting": "15%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['quality_score']}%",
                    "Metric Score": latest['quality_score'],
                    "Balanced Score": latest['quality_score'] * 0.15
                },
                {
                    "Key Performance Indicator": "Product Knowledge",
                    "Weighting": "10%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['product_knowledge']}%",
                    "Metric Score": latest['product_knowledge'],
                    "Balanced Score": latest['product_knowledge'] * 0.10
                },
                {
                    "Key Performance Indicator": "Contact Success Rate",
                    "Weighting": "10%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['contact_success_rate']}%",
                    "Metric Score": latest['contact_success_rate'],
                    "Balanced Score": latest['contact_success_rate'] * 0.10
                },
                {
                    "Key Performance Indicator": "Onboarding",
                    "Weighting": "5%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['onboarding']}%",
                    "Metric Score": latest['onboarding'],
                    "Balanced Score": latest['onboarding'] * 0.05
                },
                {
                    "Key Performance Indicator": "Reporting",
                    "Weighting": "5%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['reporting']}%",
                    "Metric Score": latest['reporting'],
                    "Balanced Score": latest['reporting'] * 0.05
                },
                {
                    "Key Performance Indicator": "Talk Time",
                    "Weighting": "10%",
                    "Best Case": 200,
                    "Worst Case": 600,
                    "Actual Performance": f"{latest['talk_time']} sec",
                    "Metric Score": max(0, min(100, ((600 - latest['talk_time']) / (600 - 200)) * 100)),
                    "Balanced Score": max(0, min(100, ((600 - latest['talk_time']) / (600 - 200)) * 100)) * 0.10
                },
                {
                    "Key Performance Indicator": "Resolution Rate",
                    "Weighting": "10%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['resolution_rate']}%",
                    "Metric Score": latest['resolution_rate'],
                    "Balanced Score": latest['resolution_rate'] * 0.10
                },
                {
                    "Key Performance Indicator": "Average Handle Time (AHT)",
                    "Weighting": "10%",
                    "Best Case": 300,
                    "Worst Case": 1200,
                    "Actual Performance": f"{latest['aht']} sec",
                    "Metric Score": max(0, min(100, ((1200 - latest['aht']) / (1200 - 300)) * 100)),
                    "Balanced Score": max(0, min(100, ((1200 - latest['aht']) / (1200 - 300)) * 100)) * 0.10
                },
                {
                    "Key Performance Indicator": "Customer Satisfaction (CSAT)",
                    "Weighting": "10%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['csat']}%",
                    "Metric Score": latest['csat'],
                    "Balanced Score": latest['csat'] * 0.10
                },
                {
                    "Key Performance Indicator": "Call Volume",
                    "Weighting": "5%",
                    "Best Case": 100,
                    "Worst Case": 0,
                    "Actual Performance": f"{latest['call_volume']} calls",
                    "Metric Score": min(100, (latest['call_volume'] / 100) * 100),
                    "Balanced Score": min(100, (latest['call_volume'] / 100) * 100) * 0.05
                }
            ]
            
            # Create a DataFrame for the table
            metrics_df = pd.DataFrame(metrics_data)
            
            # Calculate the total balanced score
            total_balanced_score = metrics_df["Balanced Score"].sum()
            
            # Add a total row
            total_row = pd.DataFrame([{
                "Key Performance Indicator": "TOTAL",
                "Weighting": "100%",
                "Best Case": "",
                "Worst Case": "",
                "Actual Performance": "",
                "Metric Score": "",
                "Balanced Score": f"{total_balanced_score:.1f}%"
            }])
            
            # Concatenate the total row to the DataFrame
            metrics_df = pd.concat([metrics_df, total_row], ignore_index=True)
            
            # Format the columns for display
            metrics_df["Performance Range"] = metrics_df.apply(
                lambda row: f"{row['Best Case']} - {row['Worst Case']}" if row['Best Case'] != "" else "", axis=1
            )
            metrics_df["Metric Score"] = metrics_df["Metric Score"].apply(
                lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
            )
            metrics_df["Balanced Score"] = metrics_df["Balanced Score"].apply(
                lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
            )
            
            # Reorder and select columns to match the image format
            display_df = metrics_df[[
                "Key Performance Indicator",
                "Weighting",
                "Performance Range",
                "Actual Performance",
                "Metric Score",
                "Balanced Score"
            ]]
            
            # Display the table
            st.subheader("Performance Scorecard")
            st.dataframe(display_df, use_container_width=True)
            
            # Show full history
            st.subheader("Your Performance History")
            st.dataframe(results)
            
            try:
                # Monthly Performance over time with comparison to all agents
                st.subheader("Your Monthly Score Over Time vs All Agents")
                if not all_performance_df.empty:
                    combined_results = pd.concat([
                        monthly_results[['date', 'overall_score', 'agent_email']],
                        all_monthly_results[['date', 'overall_score', 'agent_email']]
                    ])
                    fig = px.line(
                        combined_results,
                        x='date',
                        y='overall_score',
                        color='agent_email',
                        title="Your Monthly Score vs All Agents Average",
                        labels={'overall_score': 'Score (%)', 'date': 'Month'},
                        line_dash_map={'All Agents (Average)': 'dash'}
                    )
                else:
                    fig = px.line(
                        monthly_results,
                        x='date',
                        y='overall_score',
                        color='agent_email',
                        title="Your Monthly Score Trend",
                        labels={'overall_score': 'Score (%)', 'date': 'Month'}
                    )
                st.plotly_chart(fig)
                
                # Metrics by category
                st.subheader("Performance by Category")
                metrics_df = results[['quality_score', 'attendance', 'resolution_rate', 
                                   'product_knowledge', 'contact_success_rate']].mean().reset_index()
                metrics_df.columns = ['Metric', 'Average']
                fig2 = px.bar(metrics_df, x='Metric', y='Average', title="Your Average Metrics")
                st.plotly_chart(fig2)
                
                # Comparison Table: Individual vs All Agents (Latest Month)
                st.subheader("Your Performance vs All Agents (Latest Month)")
                if not all_performance_df.empty and not monthly_results.empty:
                    latest_month = monthly_results['date'].max()
                    individual_latest = monthly_results[monthly_results['date'] == latest_month]
                    all_latest = all_monthly_results[all_monthly_results['date'] == latest_month]
                    
                    if not individual_latest.empty and not all_latest.empty:
                        comparison_df = pd.DataFrame({
                            'Metric': ['Overall Score'],
                            'Your Score (%)': [individual_latest['overall_score'].iloc[0]],
                            'All Agents Average (%)': [all_latest['overall_score'].iloc[0]]
                        })
                        st.dataframe(comparison_df)
                    else:
                        st.info("Not enough data for latest month comparison.")
                else:
                    st.info("No data available for comparison with other agents.")
                
            except Exception as e:
                st.error(f"Error plotting data: {str(e)}")
                st.write("Raw data:")
                st.write(results)
        else:
            st.info("No performance data available for you yet. Please contact your manager to ensure your performance data has been entered.")
            st.write("Debug: Check the following:")
            st.write(f"- Ensure performance data exists for {st.session_state.user} in the 'performance' table.")
            st.write("- Verify RLS policies allow you to view your own data.")
            st.write("- Confirm that the 'agent_email' in the performance table matches your email exactly.")

if __name__ == "__main__":
    main()
