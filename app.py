import streamlit as st
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from fuzzywuzzy import fuzz

# ==== PAGE CONFIG ====
st.set_page_config(page_title="Adaptive Math Tutor", layout="wide")

# ==== THEME & STYLES ====
st.markdown("""
<style>
.main, .stApp {
    background-color: #f7faff !important;
}
.block-container {
    padding: 2rem 2rem 2rem 2rem;
}
.big-header {font-size:2.3rem;font-weight:700;padding-bottom:8px;}
.subtitle {color:#556; margin-bottom:15px;}
.card {
    background:#fff;
    border-radius:16px;
    box-shadow:0 2px 8px #e3ebf6;
    padding:1.6rem 2rem;
    margin-bottom:22px;
}
.stProgress > div > div > div {
    background-color: #4976f7 !important;
}
</style>
""", unsafe_allow_html=True)

# ==== RESOURCE LINKS ====
resource_links = {
    "Prealgebra": "https://www.khanacademy.org/math/arithmetic",
    "Number Theory": "https://artofproblemsolving.com/alcumus",
    "Counting & Probability": "https://www.khanacademy.org/math/statistics-probability/probability-library",
    "Precalculus": "https://www.khanacademy.org/math/precalculus",
    "Geometry": "https://www.khanacademy.org/math/geometry",
    "Algebra": "https://www.khanacademy.org/math/algebra",
    "Intermediate Algebra": "https://www.khanacademy.org/math/algebra2"
}

df = pd.read_csv('clean_train.csv')
with open('quiz_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# ==== HELPERS ====
def encode_level(level):
    return int(''.join(filter(str.isdigit, level)))

def encode_type(topic):
    topic_map = {t: i for i, t in enumerate(sorted(df['type'].unique()))}
    return topic_map.get(topic, 0)

def recommend_next_level(current_level, topic, problem):
    level_code = encode_level(current_level)
    type_code = encode_type(topic)
    problem_length = len(problem)
    X_input = pd.DataFrame([[level_code, type_code, problem_length]],
                           columns=['level_code','type_code','problem_length'])
    pred = clf.predict(X_input)[0]
    all_levels = sorted(df['level'].unique())
    idx = all_levels.index(current_level)
    if pred == 1 and idx < len(all_levels)-1:
        return all_levels[idx+1]
    elif pred == 0 and idx > 0:
        return all_levels[idx-1]
    return current_level

# ==== SIDEBAR ====
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3106/3106998.png", width=60)
st.sidebar.title("🧠 Math Tutor Settings")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", ["Quiz", "Analytics Dashboard"])

topics = sorted(df['type'].unique())
levels = sorted(df['level'].unique())

selected_topic = st.sidebar.selectbox("Choose Topic", topics)
selected_level = st.sidebar.selectbox("Choose Difficulty", levels)
quiz_length = st.sidebar.slider("Number of Questions", 3, 20, 10)

filtered = df[(df['type']==selected_topic)&(df['level']==selected_level)] \
    .sample(n=quiz_length, random_state=1).reset_index(drop=True)

# ==== SESSION STATE ====
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'q_ix' not in st.session_state:
    st.session_state.q_ix = 0
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'recommend' not in st.session_state:
    st.session_state.recommend = None
if 'topic_scores' not in st.session_state:
    st.session_state.topic_scores = {t: {"correct":0,"total":0} for t in df['type'].unique()}

# ===============================================================
# ======================= QUIZ PAGE =============================
# ===============================================================
if page == "Quiz":

    st.markdown('<div class="big-header">🚀 Adaptive Math Tutor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered personalized learning with real-time analytics.</div>', unsafe_allow_html=True)

    total_questions = len(filtered)
    current_question = min(st.session_state.q_ix+1, total_questions)

    with st.container():
        st.progress(current_question/total_questions)
        st.write(f"**Question {current_question} of {total_questions}**")
        st.write(f"**Score:** {st.session_state.score}")

    if st.session_state.q_ix < total_questions:
        row = filtered.iloc[st.session_state.q_ix]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"**Q{st.session_state.q_ix+1}:** {row['problem']}")
        st.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state.show_feedback:

            user_ans = st.text_input("Your Answer", key=f"ans_{st.session_state.q_ix}")

            if st.button("✔️ Submit", key=f"submit_{st.session_state.q_ix}"):

                topic = selected_topic
                expected = str(row['answer']).strip().lower()
                entered = user_ans.strip().lower()

                fuzzy_score = fuzz.ratio(entered, expected)
                correct = fuzzy_score >= 85

                if correct:
                    st.session_state.feedback = ("✅ Correct!", "success")
                    st.session_state.score += 1
                    st.session_state.topic_scores[topic]["correct"] += 1
                elif fuzzy_score > 70:
                    st.session_state.feedback = (
                        f"Almost correct! Match: {fuzzy_score}. \n\n{row['solution']}", "error")
                else:
                    st.session_state.feedback = (
                        f"❌ Incorrect.\n\n{row['solution']}", "error")

                st.session_state.topic_scores[topic]["total"] += 1

                next_level = recommend_next_level(selected_level, selected_topic, row['problem'])
                st.session_state.recommend = f"🤖 AI suggests: **{next_level}** next!"
                st.session_state.show_feedback = True
                st.rerun()

        else:
            msg, status = st.session_state.feedback
            if status=="success":
                st.success(msg)
            else:
                st.error(msg)

            if st.session_state.recommend:
                st.info(st.session_state.recommend)

            if st.button("⏭️ Next Question"):
                st.session_state.q_ix += 1
                st.session_state.feedback=None
                st.session_state.recommend=None
                st.session_state.show_feedback=False
                st.rerun()

    else:
        st.balloons()
        st.success(f"Quiz complete! Score: {st.session_state.score}/{total_questions}")

        results_row = {"timestamp":datetime.datetime.now().isoformat(),
                       "score":st.session_state.score}

        for t,res in st.session_state.topic_scores.items():
            results_row[f"{t}_correct"]=res["correct"]
            results_row[f"{t}_total"]=res["total"]

        write_header = not os.path.exists("progress.csv") or os.stat("progress.csv").st_size==0
        pd.DataFrame([results_row]).to_csv("progress.csv",mode="a",header=write_header,index=False)

        weak_topics=[t for t,res in st.session_state.topic_scores.items()
                     if res["total"]>0 and res["correct"]/res["total"]<0.5]

        if weak_topics:
            with st.expander("📚 Recommended Resources"):
                for t in weak_topics:
                    if t in resource_links:
                        st.markdown(f"- [{t} Practice]({resource_links[t]})")
        else:
            st.success("Great job — no weak topics! 🎉")

        if st.button("🔄 Restart"):
            st.session_state.score=0
            st.session_state.q_ix=0
            st.session_state.topic_scores={t:{"correct":0,"total":0} for t in df['type'].unique()}
            st.session_state.show_feedback=False
            st.rerun()

# ===============================================================
# =================== ANALYTICS DASHBOARD =======================
# ===============================================================
elif page=="Analytics Dashboard":

    st.markdown("### 📊 Analytics Dashboard")

    try:
        prog=pd.read_csv('progress.csv')

        with st.expander("See All Past Sessions"):
            st.write(prog)

        st.write("#### 🔵 Session Scores Over Time")

        prog['timestamp']=pd.to_datetime(prog['timestamp'])
        fig,ax=plt.subplots()
        ax.plot(prog['timestamp'],prog['score'],marker='o',color='#4976f7')
        ax.set_xlabel('Date')
        ax.set_ylabel('Score')
        ax.set_title('Scores vs Date')
        ax.grid(True,alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        plt.xticks(rotation=30,fontsize=9)
        st.pyplot(fig)

        topic_labels=[col.split('_')[0] for col in prog.columns if '_correct' in col]
        topic_perf=[]
        for t in topic_labels:
            correct=prog[f"{t}_correct"].sum()
            total=prog[f"{t}_total"].sum()
            pct=(correct/total*100) if total>0 else 0
            topic_perf.append((t,pct))

        if topic_perf:
            st.write("#### 🟣 Aggregate Topic Accuracy")
            topics_dash,pcts_dash=zip(*topic_perf)
            fig2,ax2=plt.subplots()
            bars=ax2.bar(topics_dash,pcts_dash,color='#f78434')
            ax2.set_ylim(0,100)
            ax2.bar_label(bars,fmt="%.0f")
            plt.xticks(rotation=25,ha='right',fontsize=9)
            st.pyplot(fig2)

    except:
        st.warning("No analytics data yet. Complete a quiz session first.")