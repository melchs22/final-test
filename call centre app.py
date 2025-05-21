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
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# [Existing imports and functions remain unchanged, e.g., init_supabase, check_db, etc.]

# Generate PDF report
def generate_pdf_report(supabase, agents, start_date, end_date, metrics):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    title_style = ParagraphStyle(name='Title', fontSize=16, leading=20, alignment=1, spaceAfter=20)
    subtitle_style = ParagraphStyle(name='Subtitle', fontSize=12, leading=15, spaceAfter=10)
    normal_style = styles['Normal']

    # Header
    elements.append(Paragraph("Call Center Agent Performance Report", title_style))
    elements.append(Paragraph(f"Date Range: {start_date} to {end_date}", normal_style))
    elements.append(Spacer(1, 12))

    for agent in agents:
        # Agent Section
        elements.append(Paragraph(f"Agent: {agent}", subtitle_style))
        elements.append(Spacer(1, 12))

        # Performance Metrics
        perf_df = get_performance(supabase, agent)
        if not perf_df.empty:
            perf_df['date'] = pd.to_datetime(perf_df['date'])
            perf_df = perf_df[(perf_df['date'] >= pd.to_datetime(start_date)) & (perf_df['date'] <= pd.to_datetime(end_date))]
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

        # Goals
        goals_df = pd.DataFrame(supabase.table("goals").select("*").eq("agent_name", agent).execute().data)
        if not goals_df.empty:
            elements.append(Paragraph("Goals", normal_style))
            for _, goal in goals_df.iterrows():
                status = goal['status']
                target = f"{goal['target_value']:.1f}{' sec' if goal['metric'] == 'aht' else ''}"
                elements.append(Paragraph(f"{goal['metric'].replace('_', ' ').title()}: Target {target}, Status: {status}", normal_style))
            elements.append(Spacer(1, 12))

        # Badges
        badges_df = pd.DataFrame(supabase.table("badges").select("*").eq("agent_name", agent).execute().data)
        if not badges_df.empty:
            elements.append(Paragraph("Badges", normal_style))
            for _, badge in badges_df.iterrows():
                elements.append(Paragraph(f"{badge['badge_name']}: {badge['description']} (Earned on {badge['earned_at'][:10]})", normal_style))
            elements.append(Spacer(1, 12))

        # Feedback
        feedback_df = get_feedback(supabase, agent)
        if not feedback_df.empty:
            feedback_df = feedback_df[pd.to_datetime(feedback_df['created_at']).between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
            if not feedback_df.empty:
                elements.append(Paragraph("Feedback", normal_style))
                for _, feedback in feedback_df.iterrows():
                    elements.append(Paragraph(f"Feedback: {feedback['message']} (Submitted on {feedback['created_at'][:10]})", normal_style))
                    if pd.notnull(feedback['manager_response']):
                        elements.append(Paragraph(f"Response: {feedback['manager_response']} (Responded on {feedback['response_timestamp'][:10]})", normal_style))
                elements.append(Spacer(1, 12))

        elements.append(Spacer(1, 20))

    # Footer
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawString(50, 30, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | BodaBoda Union")
        canvas.restoreState()

    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer

# [Existing functions like get_performance, get_feedback, etc., remain unchanged]

# Main application
def main():
    st.set_page_config(page_title="Call Center Assessment System", layout="wide")
    
    # [Existing theme setup and session state initialization unchanged]

    # Initialize Supabase
    try:
        supabase = init_supabase()
        if not check_db(supabase):
            st.error("Critical database tables are missing.")
            st.stop()
        global auth
        auth = supabase.auth
        st.session_state.supabase = supabase
    except Exception:
        st.error("Failed to connect to Supabase.")
        st.stop()

    # [Existing login and sidebar logic unchanged]

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
        
        # [Other tabs unchanged]
        
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
                
                # Custom Report Generation
                st.subheader("Generate Custom Report")
                with st.form("custom_report_form"):
                    agents = st.multiselect("Select Agents", results['agent_name'].unique(), default=results['agent_name'].unique())
                    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
                    end_date = st.date_input("End Date", value=datetime.now())
                    available_metrics = ['attendance', 'quality_score', 'product_knowledge', 'contact_success_rate',
                                        'onboarding', 'reporting', 'talk_time', 'resolution_rate', 'aht', 'csat', 'call_volume']
                    selected_metrics = st.multiselect("Select Metrics", available_metrics, default=['attendance', 'quality_score', 'csat', 'aht'])
                    if st.form_submit_button("Generate PDF Report"):
                        if agents and selected_metrics and start_date <= end_date:
                            pdf_buffer = generate_pdf_report(supabase, agents, start_date, end_date, selected_metrics)
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"agent_performance_report_{start_date}_to_{end_date}.pdf",
                                mime="application/pdf"
                            )
                            st.success("PDF report generated successfully!")
                        else:
                            st.error("Please select at least one agent, one metric, and ensure the date range is valid.")

        # [Remaining Manager Dashboard tabs unchanged]

    # [Agent Dashboard unchanged]

if __name__ == "__main__":
    main()
