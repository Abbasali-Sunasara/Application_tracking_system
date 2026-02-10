import streamlit as st
from modules.parser import extract_text_from_pdf
# NEW: Import our new analyzer functions
from modules.analyzer import calculate_similarity, extract_skills

st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")
st.title("📄 ATS Resume Analyzer")
st.markdown("Upload a Resume and Job Description to see the match.")

# 1. Create the Input Section
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Resume")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    st.header("2. Upload Job Description")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

# 2. The "Action" Button
if st.button("Analyze Match"):
    if resume_file and jd_file:
        with st.spinner("Analyzing..."):
            # A. Extract Text (The Reader)
            resume_text = extract_text_from_pdf(resume_file)
            jd_text = extract_text_from_pdf(jd_file)

            # B. Analyze (The Brain)
            score = calculate_similarity(resume_text, jd_text)
            
            # C. Display Results
            st.markdown("---")
            st.subheader("Match Result")
            
            # Create a big progress bar
            st.progress(int(score))
            st.metric(label="Match Confidence", value=f"{score}%")
            
            # Logic for feedback
            if score > 75:
                st.success("✅ High Match! This candidate looks promising.")
            elif score > 50:
                st.warning("⚠️ Medium Match. Check for missing skills.")
            else:
                st.error("❌ Low Match. Significant skills missing.")

            # Show extracted skills side-by-side
            st.markdown("---")
            st.subheader("Keyword Analysis")
            
            # Get the lists of skills (We lowercase them to avoid 'Python' vs 'python')
            res_skills = {s.lower() for s in extract_skills(resume_text)}
            jd_skills = {s.lower() for s in extract_skills(jd_text)}
            
            # Intersection (Match) and Difference (Missing)
            common_skills = list(res_skills.intersection(jd_skills))
            missing_skills = list(jd_skills - res_skills)
            
            # Displaying them in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"✅ Matched Skills ({len(common_skills)})")
                if common_skills:
                    # Added 'color: #000000 !important' to force black text
                    st.markdown("".join([f"<span style='background-color:#008000; color:#000000 !important; padding:5px; margin:2px; border-radius:5px; display:inline-block; font-weight:bold;'>{skill}</span>" for skill in common_skills]), unsafe_allow_html=True)
                else:
                    st.write("No direct matches found.")
                
            with col2:
                st.error(f"⚠️ Missing Skills ({len(missing_skills)})")
                if missing_skills:
                    # Added 'color: #000000 !important' to force black text
                    st.markdown("".join([f"<span style='background-color:#ff0000; color:#000000 !important; padding:5px; margin:2px; border-radius:5px; display:inline-block; font-weight:bold;'>{skill}</span>" for skill in missing_skills]), unsafe_allow_html=True)
                else:
                    st.write("No missing skills found!")
            with st.expander("View Text Details"):
                st.text_area("Resume Text", resume_text, height=100)
                st.text_area("JD Text", jd_text, height=100)
                
    else:
        st.warning("Please upload both files first.")