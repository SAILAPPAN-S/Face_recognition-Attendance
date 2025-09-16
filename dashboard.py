"""
Dashboard Module for Smart Attendance System
Web interface using Streamlit for system management and monitoring

Author: AI Assistant
Date: 2025
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime, date, timedelta
import pandas as pd
import time
import os
import base64
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import custom modules
from face_recognition_system import FaceRecognitionSystem, CameraManager
from chatbot import render_chatbot
from database import DatabaseManager, ReportGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, responsive styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 20px;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 25px;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 15px 0;
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .info-card {
        background: linear-gradient(145deg, #e3f2fd, #f3e5f5);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.1);
    }
    
    .success-box {
        background: linear-gradient(145deg, #e8f5e8, #f1f8e9);
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #2e7d32;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(145deg, #fff8e1, #fffbf0);
        border: 1px solid #ff9800;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #f57c00;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.1);
    }
    
    .error-box {
        background: linear-gradient(145deg, #ffebee, #fce4ec);
        border: 1px solid #f44336;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #c62828;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.1);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Camera Feed Styles */
    .camera-container {
        background: #000;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    
    /* Student Card Styles */
    .student-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .student-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Progress Bar Styles */
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        margin: 5px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #4caf50, #8bc34a);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 15px;
            margin: 10px 0;
        }
        
        .student-card {
            padding: 15px;
        }
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8, #6a4190);
    }
</style>
""", unsafe_allow_html=True)

class AttendanceApp:
    def __init__(self):
        """Initialize the Attendance App"""
        self.db_manager = DatabaseManager()
        self.report_generator = ReportGenerator(self.db_manager)
        self.face_recognition = None
        self.camera_manager = None
    
    def show_dashboard(self):
        """Display the main dashboard"""
        st.markdown('<h1 class="main-header">📹 Smart Attendance System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Face Recognition for Seamless Attendance Management</p>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            page = st.selectbox(
                "Choose a page:",
                ["🏠 Welcome", "📊 Dashboard", "📷 Live Attendance", "👥 Manage Students", "📊 Reports", "🤖 Chatbot", "⚙️ Settings"],
                label_visibility="collapsed"
            )
            
            # Add system status
            st.markdown("---")
            st.markdown("### 📊 System Status")
            
            # Check system health
            db_status = "🟢 Online" if os.path.exists("data/attendance.db") else "🔴 Offline"
            faces_status = "🟢 Ready" if os.path.exists("data/known_faces") else "🔴 Not Ready"
            
            st.markdown(f"**Database:** {db_status}")
            st.markdown(f"**Face Recognition:** {faces_status}")
            
            # Quick stats
            try:
                stats = self.db_manager.get_attendance_statistics()
                st.metric("Total Students", stats.get('total_students', 0))
                st.metric("Present Today", stats.get('students_present_today', 0))
            except:
                st.metric("Total Students", 0)
                st.metric("Present Today", 0)
        
        # Route to different pages
        if page == "🏠 Welcome":
            self.show_landing_page()
        elif page == "📊 Dashboard":
            self.show_home_page()
        elif page == "📷 Live Attendance":
            self.show_live_attendance()
        elif page == "👥 Manage Students":
            self.show_manage_students()
        elif page == "📊 Reports":
            self.show_reports()
        elif page == "🤖 Chatbot":
            render_chatbot()
        elif page == "⚙️ Settings":
            self.show_settings()
    
    def show_landing_page(self):
        """Display an attractive landing page with system overview"""
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0; color: white;">
            <h1 style="font-size: 3.5rem; margin: 0; font-weight: 700;">📹 Smart Attendance System</h1>
            <p style="font-size: 1.3rem; margin: 20px 0; opacity: 0.9;">AI-Powered Face Recognition for Seamless Attendance Management</p>
            <div style="display: flex; justify-content: center; gap: 30px; margin-top: 30px; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">🎯</div>
                    <div style="font-weight: 600;">Easy Setup</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">⚡</div>
                    <div style="font-weight: 600;">Real-time Recognition</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">📊</div>
                    <div style="font-weight: 600;">Detailed Analytics</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("## ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #667eea; margin-top: 0;">🎯 Face Recognition</h3>
                <p>Advanced AI-powered face recognition technology for accurate attendance tracking.</p>
                <ul>
                    <li>High accuracy recognition</li>
                    <li>Liveness detection</li>
                    <li>Multiple face support</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #667eea; margin-top: 0;">📊 Analytics & Reports</h3>
                <p>Comprehensive reporting and analytics for attendance management.</p>
                <ul>
                    <li>Daily/Monthly reports</li>
                    <li>Attendance trends</li>
                    <li>Export capabilities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h3 style="color: #667eea; margin-top: 0;">🤖 Smart Assistant</h3>
                <p>AI-powered chatbot for instant attendance queries and support.</p>
                <ul>
                    <li>Natural language queries</li>
                    <li>Real-time data access</li>
                    <li>Instant responses</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick start section
        st.markdown("---")
        st.markdown("## 🚀 Quick Start Guide")
        
        steps = [
            ("1️⃣", "Add Students", "Upload student photos and information to the system"),
            ("2️⃣", "Train System", "Let the AI learn and recognize student faces"),
            ("3️⃣", "Mark Attendance", "Use live camera to automatically mark attendance"),
            ("4️⃣", "View Reports", "Generate detailed reports and analytics")
        ]
        
        for i, (icon, title, description) in enumerate(steps):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"<div style='text-align: center; font-size: 2rem;'>{icon}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{title}**")
                st.markdown(f"{description}")
            if i < len(steps) - 1:
                st.markdown("---")
    
    def show_home_page(self):
        """Display the home dashboard page"""
        st.markdown("## 📊 Dashboard Overview")
        
        # Get today's statistics
        today = date.today()
        stats = self.db_manager.get_attendance_statistics()
        today_attendance = self.db_manager.get_daily_attendance(today)
        
        # Display metrics with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0; font-size: 2.5rem;">{}</h3>
                <p style="color: #6c757d; margin: 5px 0 0 0; font-size: 1.1rem;">📚 Total Students</p>
            </div>
            """.format(stats.get('total_students', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #4caf50; margin: 0; font-size: 2.5rem;">{}</h3>
                <p style="color: #6c757d; margin: 5px 0 0 0; font-size: 1.1rem;">✅ Present Today</p>
            </div>
            """.format(len(today_attendance)), unsafe_allow_html=True)
        
        with col3:
            absent_count = max(0, stats.get('total_students', 0) - len(today_attendance))
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #f44336; margin: 0; font-size: 2.5rem;">{}</h3>
                <p style="color: #6c757d; margin: 5px 0 0 0; font-size: 1.1rem;">❌ Absent Today</p>
            </div>
            """.format(absent_count), unsafe_allow_html=True)
        
        with col4:
            attendance_rate = (len(today_attendance) / max(1, stats.get('total_students', 1))) * 100
            rate_color = "#4caf50" if attendance_rate >= 75 else "#ff9800" if attendance_rate >= 50 else "#f44336"
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: {}; margin: 0; font-size: 2.5rem;">{:.1f}%</h3>
                <p style="color: #6c757d; margin: 5px 0 0 0; font-size: 1.1rem;">📈 Attendance Rate</p>
            </div>
            """.format(rate_color, attendance_rate), unsafe_allow_html=True)
        
        # Add visual chart
        st.markdown("---")
        st.markdown("## 📈 Attendance Trends")
        
        # Create attendance trend chart
        try:
            # Get last 7 days data
            end_date = today
            start_date = today - timedelta(days=6)
            
            attendance_trend = []
            for i in range(7):
                check_date = start_date + timedelta(days=i)
                daily_attendance = self.db_manager.get_daily_attendance(check_date)
                attendance_trend.append({
                    'date': check_date.strftime('%Y-%m-%d'),
                    'present': len(daily_attendance),
                    'total': stats.get('total_students', 0)
                })
            
            if attendance_trend:
                df_trend = pd.DataFrame(attendance_trend)
                df_trend['attendance_rate'] = (df_trend['present'] / df_trend['total'] * 100).round(1)
                
                # Create line chart
                fig = px.line(
                    df_trend, 
                    x='date', 
                    y='attendance_rate',
                    title='7-Day Attendance Rate Trend',
                    labels={'attendance_rate': 'Attendance Rate (%)', 'date': 'Date'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif"),
                    height=400
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Chart data not available yet")

        # Recent activity with enhanced styling
        st.markdown("---")
        st.markdown("## 🕒 Recent Activity")
        
        try:
            recent_attendance = self.db_manager.get_recent_attendance(10)
            if recent_attendance:
                for i, record in enumerate(recent_attendance):
                    time_ago = datetime.now() - datetime.strptime(f"{record['date']} {record['time']}", '%Y-%m-%d %H:%M:%S')
                    time_str = "Just now" if time_ago.total_seconds() < 60 else f"{int(time_ago.total_seconds()/60)}m ago"
                    
                    st.markdown(f"""
                    <div class="info-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>✅ {record['student_name']}</strong><br>
                                <small style="color: #6c757d;">{record['time']} • {record['date']}</small>
                            </div>
                            <div style="color: #667eea; font-size: 0.9rem;">{time_str}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card">
                    <p style="margin: 0; text-align: center; color: #6c757d;">
                        📝 No recent attendance records
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="info-card">
                <p style="margin: 0; text-align: center; color: #6c757d;">
                    📝 No recent attendance records
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Quick actions with enhanced styling
        st.markdown("---")
        st.markdown("## ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Mark All Present", use_container_width=True, help="Mark all enrolled students as present for today"):
                students = self.db_manager.get_all_students()
                marked = 0
                for s in students:
                    try:
                        if self.db_manager.mark_attendance(s['name'], 0.99):
                            marked += 1
                    except Exception:
                        pass
                if marked > 0:
                    st.success(f"✅ Marked {marked} students present today")
                else:
                    st.info("ℹ️ No new students were marked (already present or none enrolled)")
        
        with col2:
            if st.button("📊 Generate Report", use_container_width=True, help="Generate today's attendance report"):
                st.info("Navigate to Reports section to generate detailed reports")
        
        with col3:
            if st.button("👥 Add Student", use_container_width=True, help="Add a new student to the system"):
                st.info("Navigate to Manage Students section to add new students")
    
    def show_live_attendance(self):
        """Display live attendance capture page"""
        st.markdown("## 📷 Live Attendance Capture")
        st.markdown("Use your camera to automatically mark attendance using face recognition")
        
        # Camera controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🎥 Start Camera", key="start_camera", use_container_width=True):
                if self.initialize_camera_system():
                    st.session_state.camera_active = True
                    st.success("✅ Camera started successfully!")
                else:
                    st.error("❌ Failed to start camera. Check camera connection.")
        
        with col2:
            if st.button("⏹️ Stop Camera", key="stop_camera", use_container_width=True):
                self.cleanup_camera_system()
                st.session_state.camera_active = False
                st.success("✅ Camera stopped!")
        
        with col3:
            # Recognition settings
            st.markdown("**Recognition Settings:**")
            recognition_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05, key="live_threshold")
        
        # Camera feed with enhanced styling
        if st.session_state.get('camera_active', False):
            st.markdown("---")
            st.markdown("### 📹 Live Camera Feed")
            
            # Status indicators
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **Camera:** Active")
            with col2:
                st.markdown("🟢 **Face Detection:** Ready")
            with col3:
                st.markdown("🟢 **Recognition:** Ready")
            
            # Camera container
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            camera_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recognition status
            recognition_status = st.empty()
            
            # Simulate camera feed (in real implementation, this would be actual camera feed)
            if self.camera_manager and self.face_recognition:
                # Process a short burst of frames per run to keep Streamlit responsive
                for _ in range(60):
                    if not st.session_state.get('camera_active', False):
                        break
                    frame = self.camera_manager.get_frame()
                    if frame is None:
                        camera_placeholder.warning("No camera feed available")
                        break
                    processed_frame, recognized = self.face_recognition.process_frame(frame)
                    camera_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                    
                    # Handle recognitions with throttling
                    if recognized:
                        # Mark first recognized and stop loop
                        name, score = recognized[0]
                        recognition_status.info(f"🔍 **Recognized:** {name} (Confidence: {score:.2f})")
                        
                        # Use demo date if provided
                        override_date = st.session_state.get('demo_date', '').strip()
                        if self.db_manager.mark_attendance(name, float(score), target_date=override_date if override_date else None):
                            st.session_state.last_recognition_time[name] = datetime.now()
                            st.success(f"✅ **Attendance marked for {name}**")
                            
                            # Show success animation
                            st.balloons()
                        else:
                            st.warning(f"⚠️ **Attendance already marked for {name}**")
                        
                        # Stop the camera after success
                        self.cleanup_camera_system()
                        st.session_state.camera_active = False
                        break
                    else:
                        recognition_status.info("👀 **Looking for faces...**")
                    
                    time.sleep(0.05)
            else:
                camera_placeholder.warning("Camera not initialized")
        else:
            # Instructions when camera is not active
            st.markdown("---")
            st.markdown("### 📋 Instructions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h4 style="margin-top: 0; color: #667eea;">🎯 How to Use:</h4>
                    <ol style="margin: 10px 0; padding-left: 20px;">
                        <li>Click <strong>"Start Camera"</strong> to begin</li>
                        <li>Position your face in front of the camera</li>
                        <li>Wait for face recognition to process</li>
                        <li>Attendance will be marked automatically</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h4 style="margin-top: 0; color: #667eea;">💡 Tips:</h4>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Ensure good lighting</li>
                        <li>Look directly at the camera</li>
                        <li>Keep your face centered</li>
                        <li>Remove glasses if needed</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    def show_manage_students(self):
        """Display student management page"""
        st.markdown("## 👥 Student Management")
        st.markdown("Add, manage, and train students for face recognition attendance")
        
        # Add new student section
        st.markdown("### ➕ Add New Student")
        
        with st.form("add_student"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("👤 Student Name:", placeholder="Enter full name")
                email = st.text_input("📧 Email (optional):", placeholder="student@example.com")
            
            with col2:
                phone = st.text_input("📱 Phone (optional):", placeholder="+1 (555) 123-4567")
                face_image = st.file_uploader("📷 Upload Face Image:", type=['jpg', 'jpeg', 'png'], help="Upload a clear photo of the student's face")
            
            if st.form_submit_button("➕ Add Student", use_container_width=True):
                if name and face_image:
                    # Save uploaded image temporarily
                    temp_path = f"data/temp/{name}_{int(time.time())}.jpg"
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(face_image.getbuffer())
                    
                    # Add to database
                    if self.db_manager.add_student(name, email, phone):
                        # Add face to recognition system
                        if self.face_recognition is None:
                            self.face_recognition = FaceRecognitionSystem()
                        
                        if self.face_recognition.add_new_face(name, temp_path):
                            st.success(f"✅ **Student '{name}' added successfully!**")
                            # Clean up temp file
                            os.remove(temp_path)
                            # Mark attendance immediately for first-time training
                            try:
                                if self.db_manager.mark_attendance(name, 0.99):
                                    st.success(f"✅ **Attendance marked for {name}**")
                            except Exception:
                                pass
                        else:
                            st.error("❌ **Failed to process face image. Please try with a clearer image.**")
                    else:
                        st.error("❌ **Failed to add student. Student may already exist.**")
                        os.remove(temp_path)
                else:
                    st.error("❌ **Please fill in the student name and upload a face image.**")
        
        st.divider()
        
        # Live enrollment from camera
        st.subheader("📷 Live Enroll & Train (Camera)")
        with st.form("live_enroll_form"):
            live_name = st.text_input("Student Name (for live capture):", key="live_name")
            live_samples = st.slider("Number of samples", min_value=5, max_value=30, value=15)
            camera_index = st.number_input("Camera Index", value=0, min_value=0, max_value=10, step=1, key="enroll_cam_idx")
            if st.form_submit_button("🎓 Capture & Train from Camera"):
                if live_name:
                    # Ensure face system exists
                    if self.face_recognition is None:
                        self.face_recognition = FaceRecognitionSystem()
                    # Ensure database record exists
                    if not any(s['name'] == live_name for s in self.db_manager.get_all_students()):
                        self.db_manager.add_student(live_name)
                    # Enroll from camera
                    ok = self.face_recognition.enroll_from_camera(live_name, num_samples=live_samples, camera_index=camera_index)
                    if ok:
                        st.success(f"✅ Live enrollment completed for {live_name}")
                        # Train LBPH model for stronger runtime recognition
                        try:
                            if self.face_recognition.train_lbph():
                                st.success("LBPH model trained")
                        except Exception:
                            pass
                        # Do not auto-mark here; marking happens in Live Attendance
                        st.info("Enrollment complete. Use Live Attendance to mark presence.")
                    else:
                        st.error("❌ Live enrollment failed. Ensure camera and lighting are OK.")
                else:
                    st.error("❌ Please enter the student's name for live capture.")

        st.divider()

        # Display existing students
        st.markdown("---")
        st.markdown("### 📋 Existing Students")
        
        students = self.db_manager.get_all_students()
        
        if students:
            # Display in a nice format with enhanced styling
            for i, student in enumerate(students):
                # Get student's attendance summary
                summary = self.db_manager.get_student_attendance_summary(student['name'])
                attendance_rate = summary.get('attendance_percentage', 0)
                
                # Determine attendance status color
                if attendance_rate >= 75:
                    status_color = "#4caf50"
                    status_icon = "🟢"
                elif attendance_rate >= 50:
                    status_color = "#ff9800"
                    status_icon = "🟡"
                else:
                    status_color = "#f44336"
                    status_icon = "🔴"
                
                st.markdown(f"""
                <div class="student-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div>
                            <h3 style="margin: 0; color: #333; font-size: 1.3rem;">👤 {student['name']}</h3>
                            <p style="margin: 5px 0 0 0; color: #6c757d; font-size: 0.9rem;">
                                📧 {student.get('email', 'No email')} • 📱 {student.get('phone', 'No phone')}
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {status_color}; font-size: 1.1rem; font-weight: 600;">
                                {status_icon} {attendance_rate:.1f}%
                            </div>
                            <div style="color: #6c757d; font-size: 0.8rem;">Attendance Rate</div>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #6c757d; font-size: 0.9rem;">Attendance Progress</span>
                            <span style="color: {status_color}; font-weight: 600;">{attendance_rate:.1f}%</span>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {attendance_rate}%; background: linear-gradient(90deg, {status_color}, {status_color}88);"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("🗑️ Delete", key=f"del_{i}", use_container_width=True, help="Delete student and all related data"):
                        # Remove from recognition system first
                        if self.face_recognition is None:
                            self.face_recognition = FaceRecognitionSystem()
                        fr_deleted = self.face_recognition.remove_face(student['name'])
                        db_deleted = self.db_manager.delete_student(student['name'])
                        if db_deleted:
                            st.success(f"✅ **Deleted {student['name']} and related data**")
                            st.rerun()
                        else:
                            st.warning("⚠️ **Delete failed or student not found**")
                
                with col2:
                    if st.button("📷 Retrain", key=f"retrain_{i}", use_container_width=True, help="Retrain face recognition from camera"):
                        if self.face_recognition is None:
                            self.face_recognition = FaceRecognitionSystem()
                        # Optional: clear existing features for cleaner retrain
                        self.face_recognition.remove_face(student['name'])
                        if self.face_recognition.enroll_from_camera(student['name']):
                            st.success(f"✅ **Retrained {student['name']} from camera**")
                            if self.db_manager.mark_attendance(student['name'], 0.98):
                                st.success(f"✅ **Attendance marked for {student['name']}**")
                        else:
                            st.error("❌ **Retrain failed**")
                
                with col3:
                    if st.button("📊 View Stats", key=f"stats_{i}", use_container_width=True, help="View detailed attendance statistics"):
                        st.info(f"Navigate to Reports section to view detailed statistics for {student['name']}")
                
                if i < len(students) - 1:
                    st.markdown("---")
        else:
            st.markdown("""
            <div class="info-card">
                <div style="text-align: center; padding: 20px;">
                    <h3 style="color: #667eea; margin-bottom: 10px;">👥 No Students Yet</h3>
                    <p style="color: #6c757d; margin: 0;">Add your first student using the form above to get started!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def show_reports(self):
        """Display reports page"""
        st.header("📊 Reports & Analytics")
        
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type:",
            ["Daily Report", "Monthly Report", "Student Summary", "Attendance Statistics"]
        )
        
        if report_type == "Daily Report":
            self.show_daily_report()
        elif report_type == "Monthly Report":
            self.show_monthly_report()
        elif report_type == "Student Summary":
            self.show_student_summary()
        elif report_type == "Attendance Statistics":
            self.show_attendance_statistics()
    
    def show_daily_report(self):
        """Show daily report section"""
        st.subheader("📅 Daily Attendance Report")
        
        # Date selection
        selected_date = st.date_input("Select Date:", value=date.today())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Report", use_container_width=True):
                self.generate_daily_report(selected_date)
        
        with col2:
            if st.button("💾 Export to CSV", use_container_width=True):
                self.export_daily_csv(selected_date)
        
        with col3:
            if st.button("📄 Export to PDF", use_container_width=True):
                self.export_daily_pdf(selected_date)
        
        # Display report data
        attendance_data = self.db_manager.get_daily_attendance(selected_date)
        
        if attendance_data:
            st.success(f"Found {len(attendance_data)} attendance records for {selected_date}")
            
            # Create DataFrame
            df = pd.DataFrame(attendance_data)
            df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.strftime('%I:%M %p')
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2f}" if x else "N/A")
            
            # Display table
            st.dataframe(
                df,
                column_config={
                    "student_name": "Student Name",
                    "date": "Date",
                    "time": "Time",
                    "confidence": "Confidence Score"
                },
                use_container_width=True
            )
            
            # Create attendance chart
            present_count = len(attendance_data)
            total_students = self.db_manager.get_attendance_statistics().get('total_students', 0)
            absent_count = max(0, total_students - present_count)
            
            # Pie chart
            fig = px.pie(
                values=[present_count, absent_count],
                names=['Present', 'Absent'],
                title=f'Attendance Overview for {selected_date}',
                color_discrete_map={'Present': '#28a745', 'Absent': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info(f"No attendance records found for {selected_date}")
    
    def show_monthly_report(self):
        """Show monthly report section"""
        st.subheader("📆 Monthly Attendance Report")
        
        # Month and year selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_year = st.selectbox("Year:", range(2020, 2030), index=5)  # Default to 2025
        
        with col2:
            selected_month = st.selectbox(
                "Month:", 
                range(1, 13),
                format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
                index=datetime.now().month - 1
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Monthly Report", use_container_width=True):
                self.generate_monthly_report(selected_year, selected_month)
        
        with col2:
            if st.button("💾 Export to Excel", use_container_width=True):
                self.export_monthly_excel(selected_year, selected_month)
        
        with col3:
            if st.button("📄 Export to PDF", use_container_width=True):
                self.export_monthly_pdf(selected_year, selected_month)
    
    def show_student_summary(self):
        """Show individual student summary"""
        st.subheader("👤 Student Attendance Summary")
        
        # Get all students
        students = self.db_manager.get_all_students()
        
        if not students:
            st.warning("No students found. Please add students first.")
            return
        
        # Student selection
        student_names = [s['name'] for s in students]
        selected_student = st.selectbox("Select Student:", student_names)
        
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date:", value=date.today() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date:", value=date.today())
        
        if st.button("📊 Generate Student Summary", use_container_width=True):
            # Get student summary
            summary = self.db_manager.get_student_attendance_summary(selected_student, start_date, end_date)
            
            if summary:
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Present Days", summary['total_present'])
                
                with col2:
                    st.metric("Absent Days", summary['total_absent'])
                
                with col3:
                    st.metric("Working Days", summary['total_working_days'])
                
                with col4:
                    percentage = summary['attendance_percentage']
                    delta_color = "normal" if percentage >= 75 else "inverse"
                    st.metric("Attendance %", f"{percentage:.1f}%", delta_color=delta_color)
            else:
                st.error("Failed to generate student summary.")
    
    def show_attendance_statistics(self):
        """Show overall attendance statistics"""
        st.subheader("📈 Attendance Statistics & Analytics")
        
        # Get overall statistics
        stats = self.db_manager.get_attendance_statistics()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", stats.get('total_students', 0))
        
        with col2:
            st.metric("Present Today", stats.get('students_present_today', 0))
        
        with col3:
            st.metric("Absent Today", stats.get('students_absent_today', 0))
        
        with col4:
            st.metric("Average Attendance", f"{stats.get('average_attendance', 0):.1f}%")
    
    def show_settings(self):
        """Show settings page"""
        st.header("⚙️ System Settings")
        
        # Camera settings
        st.subheader("📷 Camera Settings")
        
        with st.form("camera_settings"):
            camera_index = st.number_input("Camera Index", value=0, min_value=0, max_value=10)
            recognition_threshold = st.slider("Recognition Threshold", 0.1, 1.0, float(st.session_state.get('match_threshold', 0.6)), 0.05)
            liveness_detection = st.checkbox("Enable Liveness Detection", value=True)
            use_lbph = st.checkbox("Use LBPH recognizer (recommended)", value=bool(st.session_state.get('use_lbph', True)))
            lbph_conf = st.slider("LBPH max confidence (lower is stricter)", 10.0, 100.0, float(st.session_state.get('lbph_conf', 70.0)), 1.0)
            st.divider()
            st.subheader("🧪 Demo Date Override")
            demo_date = st.text_input("Override date for marking/reporting (YYYY-MM-DD)", value=st.session_state.get('demo_date', ''))
            
            if st.form_submit_button("Save Camera Settings"):
                st.session_state['camera_index'] = int(camera_index)
                st.session_state['match_threshold'] = float(recognition_threshold)
                st.session_state['use_lbph'] = bool(use_lbph)
                st.session_state['lbph_conf'] = float(lbph_conf)
                st.session_state['demo_date'] = demo_date.strip()
                st.success("Camera settings saved!")
        
        st.divider()
        
        # Database settings
        st.subheader("🗄️ Database Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Backup Database", use_container_width=True):
                self.backup_database()
        
        with col2:
            if st.button("🧹 Clean Old Logs", use_container_width=True):
                try:
                    self.db_manager.cleanup_old_logs(30)
                    st.success("Old logs cleaned up!")
                except:
                    st.warning("Cleanup method not implemented")
        
        with col3:
            if st.button("📊 Database Stats", use_container_width=True):
                self.show_database_stats()
        
        st.divider()

        # Reset dataset
        st.subheader("♻️ Reset Dataset (Start Fresh)")
        st.warning("This will delete ALL students, attendance, and known face images.")
        if st.button("⚠️ Reset All Data", use_container_width=True):
            ok_db = False
            ok_faces = False
            try:
                ok_db = self.db_manager.reset_all_data()
            except Exception:
                pass
            try:
                if self.face_recognition is None:
                    self.face_recognition = FaceRecognitionSystem()
                ok_faces = self.face_recognition.reset_known_faces()
            except Exception:
                pass
            if ok_db and ok_faces:
                st.success("All data cleared. You can now re-enroll and retrain from scratch.")
            else:
                st.warning("Some items could not be cleared. Check file permissions and try again.")
        
        # System information
        st.subheader("💻 System Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("**Application Version:** 1.0.0")
            st.write("**Database Path:** data/attendance.db")
            st.write("**Known Faces:** data/known_faces/")
        
        with info_col2:
            # Check if required files exist
            db_exists = os.path.exists("data/attendance.db")
            faces_dir_exists = os.path.exists("data/known_faces")
            
            st.write(f"**Database Status:** {'✅ OK' if db_exists else '❌ Missing'}")
            st.write(f"**Known Faces Dir:** {'✅ OK' if faces_dir_exists else '❌ Missing'}")
            
            # Count known faces
            if faces_dir_exists:
                face_files = [f for f in os.listdir("data/known_faces") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                st.write(f"**Face Images:** {len(face_files)} files")
    
    def initialize_camera_system(self):
        """Initialize camera and face recognition systems"""
        try:
            if self.face_recognition is None:
                match_th = float(st.session_state.get('match_threshold', 0.6))
                self.face_recognition = FaceRecognitionSystem(
                    match_threshold=match_th,
                    use_lbph=bool(st.session_state.get('use_lbph', True)),
                    lbph_confidence_max=float(st.session_state.get('lbph_conf', 70.0)),
                )
            
            if self.camera_manager is None:
                cam_idx = int(st.session_state.get('camera_index', 0))
                self.camera_manager = CameraManager(camera_index=cam_idx)
                if not self.camera_manager.start_camera():
                    st.error("Failed to initialize camera")
                    return False
            
            return True
        except Exception as e:
            st.error(f"Error initializing camera system: {e}")
            return False
    
    def cleanup_camera_system(self):
        """Cleanup camera and face recognition systems"""
        try:
            if self.camera_manager:
                self.camera_manager.stop_camera()
                self.camera_manager = None
        except Exception as e:
            logger.error(f"Error cleaning up camera system: {e}")
    
    def generate_daily_report(self, target_date: date):
        """Generate and display daily report"""
        try:
            report_data = self.report_generator.generate_daily_report(target_date)
            if report_data['attendance_data']:
                st.success(f"Daily report generated for {target_date}")
            else:
                st.info(f"No attendance data found for {target_date}")
        except:
            st.info(f"No attendance data found for {target_date}")
    
    def export_daily_csv(self, target_date: date):
        """Export daily report to CSV"""
        try:
            attendance_data = self.db_manager.get_daily_attendance(target_date)
            if attendance_data:
                filename = f"daily_report_{target_date}.csv"
                if self.db_manager.export_to_csv(attendance_data, filename):
                    st.success(f"Report exported to {filename}")
                    # Offer download
                    file_path = os.path.join("data", "reports", filename)
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="Download CSV",
                                data=f.read(),
                                file_name=filename,
                                mime="text/csv",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.warning(f"Unable to read exported file: {e}")
                else:
                    st.warning("Export failed")
            else:
                st.warning("No data to export")
        except:
            st.warning("Export method not implemented")
    
    def export_daily_pdf(self, target_date: date):
        """Export daily report to PDF"""
        try:
            report_data = self.report_generator.generate_daily_report(target_date)
            if report_data['attendance_data']:
                filename = f"daily_report_{target_date}.pdf"
                title = f"Daily Attendance Report - {target_date}"
                
                if self.db_manager.generate_pdf_report(
                    report_data['attendance_data'],
                    report_data['summary_data'],
                    filename,
                    title
                ):
                    st.success(f"PDF report generated: {filename}")
                    # Offer download
                    file_path = os.path.join("data", "reports", filename)
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="Download PDF",
                                data=f.read(),
                                file_name=filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.warning(f"Unable to read generated PDF: {e}")
                else:
                    st.warning("PDF generation failed")
            else:
                st.warning("No data to export")
        except:
            st.warning("PDF export method not implemented")
    
    def generate_monthly_report(self, year: int, month: int):
        """Generate monthly report"""
        try:
            report_data = self.report_generator.generate_monthly_report(year, month)
            if report_data['attendance_data']:
                st.success(f"Monthly report generated for {datetime(year, month, 1).strftime('%B %Y')}")
            else:
                st.info(f"No attendance data found for {datetime(year, month, 1).strftime('%B %Y')}")
        except:
            st.info(f"No attendance data found for {datetime(year, month, 1).strftime('%B %Y')}")
    
    def export_monthly_excel(self, year: int, month: int):
        """Export monthly report to Excel"""
        try:
            report_data = self.report_generator.generate_monthly_report(year, month)
            if report_data['attendance_data']:
                filename = f"monthly_report_{year}_{month:02d}.xlsx"
                
                if self.db_manager.export_to_excel(report_data['attendance_data'], filename):
                    st.success(f"Report exported to {filename}")
                    # Offer download
                    file_path = os.path.join("data", "reports", filename)
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="Download Excel",
                                data=f.read(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.warning(f"Unable to read exported Excel: {e}")
                else:
                    st.warning("Export failed")
            else:
                st.warning("No data to export")
        except:
            st.warning("Excel export method not implemented")
    
    def export_monthly_pdf(self, year: int, month: int):
        """Export monthly report to PDF"""
        try:
            report_data = self.report_generator.generate_monthly_report(year, month)
            if report_data['attendance_data']:
                filename = f"monthly_report_{year}_{month:02d}.pdf"
                title = f"Monthly Attendance Report - {datetime(year, month, 1).strftime('%B %Y')}"
                
                if self.db_manager.generate_pdf_report(
                    report_data['attendance_data'],
                    report_data['summary_data'],
                    filename,
                    title
                ):
                    st.success(f"PDF report generated: {filename}")
                    # Offer download
                    file_path = os.path.join("data", "reports", filename)
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="Download PDF",
                                data=f.read(),
                                file_name=filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.warning(f"Unable to read generated PDF: {e}")
                else:
                    st.warning("PDF generation failed")
            else:
                st.warning("No data to export")
        except:
            st.warning("PDF export method not implemented")
    
    def backup_database(self):
        """Create database backup"""
        try:
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"attendance_backup_{timestamp}.db"
            backup_path = os.path.join("data/reports", backup_filename)
            
            shutil.copy2(self.db_manager.db_path, backup_path)
            st.success(f"Database backed up to: {backup_filename}")
        except Exception as e:
            st.error(f"Backup failed: {e}")
    
    def show_database_stats(self):
        """Show database statistics"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Get table sizes
            cursor.execute("SELECT COUNT(*) FROM students")
            student_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM attendance")
            attendance_count = cursor.fetchone()[0]
            
            conn.close()
            
            st.info(f"""
            **Database Statistics:**
            - Students: {student_count}
            - Attendance Records: {attendance_count}
            """)
        except Exception as e:
            st.error(f"Failed to get database stats: {e}")

# Main application entry point
def main():
    """Main function to run the Streamlit app"""
    try:
        # Initialize session state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'last_recognition_time' not in st.session_state:
            st.session_state.last_recognition_time = {}
        
        app = AttendanceApp()
        app.show_dashboard()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()