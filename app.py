import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Govt Scheme Facilitator", layout="wide")
st.title("🏛️ Govt Scheme Facilitator")

# ---------------- PERSONA ----------------
prompt = ChatPromptTemplate.from_template("""
You are a Government Scheme Expert.

ONLY use the provided scheme documents.
DO NOT make assumptions.

If the answer is not found, say:
"I don't know based on available data."
""")

# ---------------- LOAD DATA ----------------
def get_data_path():
    return os.path.join(os.path.dirname(__file__), "data")

@st.cache_resource
def load_db():
    docs = []
    for file in os.listdir(get_data_path()):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(get_data_path(), file), encoding="utf-8")
            docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

db = load_db()

# ---------------- INPUT ----------------
col1, col2, col3, col4 = st.columns(4)

age = col1.number_input("Age", 0, 100)
category = col2.selectbox("Category", ["Student", "Farmer", "General"])
caste = col3.selectbox("Caste", ["General", "OBC", "SC", "ST"])
income = col4.number_input("Income (₹)", 0, 2000000, step=50000)

query = f"{category} {caste} {income} {age}"

# ---------------- PARSE ----------------
def parse_scheme(text):
    data = {
        "name": "",
        "category": "",
        "caste": [],
        "income": 99999999,
        "age": "Any",
        "link": "",
        "benefits": ""
    }

    lines = text.split("\n")
    capture = False
    benefits = []

    for line in lines:
        line = line.strip()

        if line.startswith("Scheme Name"):
            data["name"] = line.split(":",1)[1].strip()

        elif line.startswith("Category"):
            data["category"] = line.split(":",1)[1].strip()

        elif line.startswith("Caste"):
            data["caste"] = [x.strip() for x in line.split(":",1)[1].split(",")]

        elif line.startswith("Income"):
            val = line.split(":",1)[1].strip()
            if val.lower() == "any":
                data["income"] = 99999999
            else:
                val = val.replace("<=", "")
                if val.isdigit():
                    data["income"] = int(val)

        elif line.startswith("Age"):
            data["age"] = line.split(":",1)[1].strip()

        elif line.startswith("Link"):
            data["link"] = line.split(":",1)[1].strip()

        elif line.startswith("Benefits"):
            capture = True
            continue

        elif line == "":
            capture = False

        elif capture:
            benefits.append(line)

    data["benefits"] = " ".join(benefits)
    return data

# ---------------- BUTTON ----------------
if st.button("🔍 Find Schemes"):

    results = db.similarity_search(query, k=6)

    if not results:
        st.error("I don't know based on available data.")
    else:

        eligible = []
        not_eligible = []

        for doc in results:
            scheme = parse_scheme(doc.page_content)

            ok = True
            reason = []

            # Category
            if scheme["category"].lower() != category.lower():
                ok = False
                reason.append("Category mismatch")

            # Caste
            if caste not in scheme["caste"] and "All" not in scheme["caste"]:
                ok = False
                reason.append("Caste not eligible")

            # Income
            if income > scheme["income"]:
                ok = False
                reason.append("Income exceeds limit")

            # AGE LOGIC (FINAL FIX)
            if scheme["age"] != "Any":
                try:
                    low, high = map(int, scheme["age"].split("-"))
                    if not (low <= age <= high):
                        ok = False
                        reason.append(f"Age not eligible ({scheme['age']})")
                except:
                    pass

            scheme["reason"] = ", ".join(reason) if reason else "Meets all criteria"

            if ok:
                eligible.append(scheme)
            else:
                not_eligible.append(scheme)

        # -------- ELIGIBLE --------
        st.subheader("✅ Eligible Schemes")

        if not eligible:
            st.warning("I don't know based on available data.")

        for s in eligible:
            st.success(s["name"])
            st.write("Category:", s["category"])
            st.write("Caste:", ", ".join(s["caste"]))
            st.write("Age Limit:", s["age"])
            st.write("Income:", "No limit" if s["income"] == 99999999 else f"₹{s['income']}")
            st.write("Benefits:", s["benefits"])

            if s["link"]:
                st.link_button("Apply Now 🚀", s["link"])

            st.divider()

        # -------- NOT ELIGIBLE --------
        st.subheader("❌ Not Eligible Schemes")

        for s in not_eligible:
            st.error(s["name"])
            st.write("Category:", s["category"])
            st.write("Caste:", ", ".join(s["caste"]))
            st.write("Age Limit:", s["age"])
            st.write("Income:", "No limit" if s["income"] == 99999999 else f"₹{s['income']}")
            st.write("Benefits:", s["benefits"])
            st.write("Reason:", s["reason"])
            st.divider()