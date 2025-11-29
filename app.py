import streamlit as st
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Student, Teacher, DailyLog
from models.predict_model import predict_subject_outcome
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------ STREAMLIT SETUP ------------------
st.set_page_config(page_title="Student Academic Success Prediction using Machine Learning ", layout="wide")

# Custom CSS for UI Styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .header {
            font-size: 2rem;
            color: #1e3d58;
            text-align: center;
            font-weight: bold;
        }
        .dashboard-card {
            background-color: #f1f8e9;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .btn-primary {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
        }
        .btn-primary:hover {
            background-color: #0056b3;
        }
        .chart-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
        }
        .column {
            width: 45%;
            padding: 10px;
        }
        .metric-card {
            background-color: #f1f8e9;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .metric-card h3 {
            color: #4caf50;
            font-weight: bold;
        }
        .metric-card p {
            font-size: 1.5rem;
            color: #333;
        }
        .recommendation-box {
            background-color: #e0f7fa;
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            margin-top: 20px;
        }
        .recommendation-box p {
            color: #00796b;
        }
    </style>
""", unsafe_allow_html=True)

# Database setup
DB_URL = "sqlite:///student_tracker.db"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
session_db = Session()

# Create tables if not exist
Base.metadata.create_all(engine)

SUBJECTS = ["Math", "Science", "English", "History", "Geography", "Computer Science"]

# ------------------ AI-Powered Recommendations Helper Function ------------------
def get_recommendations(cluster_id):
    if cluster_id == 0:
        return "Focus more on mental health and increase sleep hours."
    elif cluster_id == 1:
        return "Increase your focus and self-study hours for better results."
    else:
        return "Maintain your current routine, but monitor sleep patterns."

# ------------------ ML MODELS ------------------

# Performance prediction model (e.g., RandomForest or XGBoost)
def subject_performance_prediction(logs, model_type="RandomForest"):
    # Prepare Data
    df = pd.DataFrame([{
        "Attendance": l.attendance,
        "Self Study": l.self_study,
        "Mental": l.mental,
        "Focus": l.focus,
        "Outcome": 1 if l.attendance >= 75 and l.focus >= 7 else 0  # Simple pass/fail prediction rule
    } for l in logs])

    X = df[["Attendance", "Self Study", "Mental", "Focus"]]
    y = df["Outcome"]

    # Train model
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Model not supported")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)

    # Predict
    prediction = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    return prediction[0], accuracy

# Clustering for AI-powered study plan recommendations
def cluster_students(logs):
    df = pd.DataFrame([{
        "Self Study": l.self_study,
        "Focus": l.focus,
        "Mental": l.mental,
        "Attendance": l.attendance
    } for l in logs])

    # Using KMeans clustering
    kmeans = KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(df)

    # Determine the cluster for the student
    student_cluster = kmeans.predict([[df["Self Study"].mean(), df["Focus"].mean(), df["Mental"].mean(), df["Attendance"].mean()]])
    return student_cluster[0]

# ------------------ AUTH HELPERS ------------------
def login_user(username, password, role):
    if role == "student":
        return session_db.query(Student).filter_by(username=username, password=password).first()
    else:
        return session_db.query(Teacher).filter_by(username=username, password=password).first()

def register_user(username, password, role):
    if role == "student":
        existing = session_db.query(Student).filter_by(username=username).first()
        if existing:
            return False, "Student username already exists."
        user = Student(username=username, password=password)
    else:
        existing = session_db.query(Teacher).filter_by(username=username).first()
        if existing:
            return False, "Teacher username already exists."
        user = Teacher(username=username, password=password)
    session_db.add(user)
    session_db.commit()
    return True, "Registered successfully!"

# ------------------ SESSION STATE ------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "refresh" not in st.session_state:
    st.session_state.refresh = False  # toggling flag to trigger rerun if needed

# ------------------ SIDEBAR LOGIN/REGISTER ------------------
st.sidebar.title("🔐 Authentication")

if not st.session_state.user:
    tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])

    # Login Tab
    with tab_login:
        role = st.radio("Login as:", ["student", "teacher"], key="login_role")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            user = login_user(username, password, role)
            if user:
                st.session_state.user = username
                st.session_state.role = role
                st.sidebar.success(f"Welcome, {username}!")
                # Toggle refresh flag to force rerun
                st.session_state.refresh = not st.session_state.refresh
            else:
                st.sidebar.error("Invalid credentials.")

    # Register Tab
    with tab_register:
        role = st.radio("Register as:", ["student", "teacher"], key="register_role")
        username = st.text_input("Choose a username", key="register_user")
        password = st.text_input("Choose a password", type="password", key="register_pass")
        if st.button("Register", key="register_btn"):
            success, msg = register_user(username, password, role)
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
else:
    def logout():
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.refresh = not st.session_state.refresh

    if st.sidebar.button("Logout"):
        logout()
        st.sidebar.info("You have been logged out.")

# ------------------ MAIN APP ------------------
st.title("🎓 Student Academic Success Prediction")

if not st.session_state.user:
    st.info("👈 Please log in or register from the sidebar to continue.")
else:
    role = st.session_state.role

    # ------------------ STUDENT DASHBOARD ------------------
    if role == "student":
        st.markdown('<div class="header">Welcome, Student Dashboard</div>', unsafe_allow_html=True)

        # Add daily log form in a card-like container
        with st.expander("➕ Add Daily Log", expanded=True):
            with st.form("daily_log_form"):
                subject = st.selectbox("Subject", SUBJECTS)
                log_date = st.date_input("Date", value=date.today())
                attendance = st.slider("Attendance (%)", 0, 100, 100)
                self_study = st.slider("Self Study (hours)", 0, 10, 2)
                mental = st.slider("Mental Wellbeing (1-10)", 1, 10, 7)
                sleep = st.number_input("Sleep (hours)", 0.0, 12.0, 7.0)
                focus = st.slider("Focus (1-10)", 1, 10, 8)
                submitted = st.form_submit_button("Add Log")

                if submitted:
                    try:
                        new_log = DailyLog(
                            student_username=st.session_state.user,
                            date=log_date,
                            subject=subject,
                            attendance=attendance,
                            self_study=self_study,
                            mental=mental,
                            sleep=sleep,
                            focus=focus
                        )
                        session_db.add(new_log)
                        session_db.commit()
                        st.success("✅ Log added successfully!")
                    except Exception as e:
                        st.error(f"Error adding log: {e}")

        # Display logs in a dashboard-like card
        st.subheader("📅 Your Logs")
        logs = session_db.query(DailyLog).filter_by(student_username=st.session_state.user).all()
        if logs:
            st.table(
                [{"Date": l.date, "Subject": l.subject, "Attendance": l.attendance,
                  "Self Study": l.self_study, "Mental": l.mental,
                  "Sleep": l.sleep, "Focus": l.focus} for l in logs]
            )
        else:
            st.info("No logs yet. Add one above!")

        # Prediction section in a visually distinct box
        st.markdown('<div class="dashboard-card"><h3>📊 Predict Subject Outcome</h3></div>', unsafe_allow_html=True)
        selected_subject = st.selectbox("Select Subject", SUBJECTS)
        if st.button("Predict"):
            logs = session_db.query(DailyLog).filter_by(student_username=st.session_state.user, subject=selected_subject).all()
            prediction, accuracy = subject_performance_prediction(logs)
            st.success(f"Prediction: **{'Pass' if prediction == 1 else 'Fail'}** (Model accuracy: {accuracy*100:.2f}%)")

        # ------------------ AI-Powered Study Plan Recommendations ------------------
        st.markdown('<div class="recommendation-box"><p>🧑‍🎓 AI-Powered Study Plan Recommendations</p></div>', unsafe_allow_html=True)

        logs = session_db.query(DailyLog).filter_by(student_username=st.session_state.user).all()
        if logs:
            student_cluster = cluster_students(logs)
            recommendation = get_recommendations(student_cluster)
            st.write(recommendation)
        else:
            st.info("No logs yet. Add one above to get recommendations.")
    
    # ------------------ TEACHER DASHBOARD ------------------
    elif role == "teacher":
        st.markdown('<div class="header">Welcome, Teacher Dashboard</div>', unsafe_allow_html=True)

        # Get all students
        students = [s.username for s in session_db.query(Student).all()]
        if not students:
            st.info("No students registered yet.")
        else:
            selected_student = st.selectbox("Select a Student", students)

            # Fetch logs for selected student
            logs = session_db.query(DailyLog).filter_by(student_username=selected_student).all()

            if logs:
                # Display logs
                st.subheader(f"📅 Daily Logs for {selected_student}")
                df = pd.DataFrame(
                    [{"Date": l.date, "Subject": l.subject, "Attendance": l.attendance,
                      "Self Study": l.self_study, "Mental": l.mental,
                      "Sleep": l.sleep, "Focus": l.focus} for l in logs]
                )
                st.dataframe(df, use_container_width=True)

                # Performance Trends Visualization
                st.subheader("📊 Performance Trends")
                df['Date'] = pd.to_datetime(df['Date'])
                df_sorted = df.sort_values('Date')
                metrics = ["Attendance", "Self Study", "Focus", "Mental"]
                
                # Plot performance trends
                fig, ax = plt.subplots(figsize=(10, 5))
                for m in metrics:
                    ax.plot(df_sorted["Date"], df_sorted[m], marker="o", label=m)
                ax.set_title(f"Performance Trends for {selected_student}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Score / Hours / %")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Focus & Study Time Trends
                st.subheader("📊 Focus & Study Time Trends Over Time")
                st.line_chart(df_sorted.set_index('Date')[["Focus", "Self Study"]])

                # Performance Prediction (based on model)
                st.subheader("📊 Predictive Scenarios")
                avg_focus = df["Focus"].mean()
                avg_self_study = df["Self Study"].mean()
                avg_attendance = df["Attendance"].mean()

                st.metric("Avg Focus", f"{avg_focus}/10")
                st.metric("Avg Self Study", f"{avg_self_study} hrs")
                st.metric("Avg Attendance", f"{avg_attendance}%")

                scenario_1 = predict_subject_outcome(logs)  # Overall model
                scenario_2 = 1 if avg_focus > 7 else 0  # Focus-based heuristic
                scenario_3 = 1 if avg_self_study > 3 else 0  # Study-time based heuristic

                # Display Scenario results
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"<div class='metric-card'>Scenario 1<br>{'✅ Pass' if scenario_1 == 1 else '❌ Fail'}</div>", unsafe_allow_html=True
                    )
                with c2:
                    st.markdown(
                        f"<div class='metric-card'>Scenario 2<br>{'✅ Pass' if scenario_2 == 1 else '❌ Fail'}</div>", unsafe_allow_html=True
                    )
                with c3:
                    st.markdown(
                        f"<div class='metric-card'>Scenario 3<br>{'✅ Pass' if scenario_3 == 1 else '❌ Fail'}</div>", unsafe_allow_html=True
                    )

            else:
                st.warning(f"No logs found for student **{selected_student}**.")
