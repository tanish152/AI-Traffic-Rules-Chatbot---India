<div align="center">

# 🚦 AI Traffic Rules Assistant  
### *AI That Understands Every Rule of the Road*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aitrafficrulechatbot.streamlit.app/)
![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Maintenance](https://img.shields.io/badge/Maintained-Yes-green.svg)

###  
[**Live Demo**](https://aitrafficrulechatbot.streamlit.app/) • 
[**Report Bug**](https://github.com/tanish152/AI-Traffic-Rules-Chatbot---India/issues) • 
[**Request Feature**](https://github.com/tanish152/AI-Traffic-Rules-Chatbot---India/issues)

</div>

---

## 📋 Table of Contents
- About the Project
- Key Features
- Screenshots
- Tech Stack
- Getting Started
- Project Structure
- How It Works
- Database Statistics
- Usage Guide
- Contributing
- License
- Contact
- Acknowledgments

---

## 🎯 About the Project

AI Traffic Rules Assistant is a smart, NLP-powered platform that helps users instantly find accurate Indian traffic rules, penalties, and regulations.  
Using TF-IDF and Cosine Similarity, the app delivers context-aware results tailored by:

✔ City  
✔ Vehicle Type  
✔ Offense Category  

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🔎 Smart Semantic Search | NLP-powered search |
| 🏙 City-Based Results | Location-specific |
| 🚗 Vehicle Type Filtering | Cars, Bikes, etc |
| 🚫 No Duplicate Rules | Optimized results |
| 🌗 Day/Night Mode | Theme toggle |
| 📊 Database Insights | Quick statistics |
| 📞 Emergency Contacts | Helpline numbers |
| 💡 Quick Suggestions | Pre-built queries |
| ⚡ High Performance | Fast + accurate |

---

## 🖼 Screenshots

### Day Mode  
![Day Mode](assets/day-mode.png)

### Night Mode  
![Night Mode](assets/night-mode.png)

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- OpenPyXL  

---

## 🚀 Getting Started

### Prerequisites

```bash
python --version
```

### Installation

```bash
git clone https://github.com/tanish152/AI-Traffic-Rules-Chatbot---India.git
cd AI-Traffic-Rules-Chatbot---India
```

### Virtual Environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### Install Packages

```bash
pip install -r requirements.txt
```

### Run App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

AI-Traffic-Rules-Chatbot---India/  
│── app.py  
│── README.md  
│── requirements.txt  
│── LICENSE  
│  
├── data/  
│   └── traffic_rules.xlsx  
│  
├── assets/  
│   ├── day-mode.png  
│   ├── night-mode.png  
│   └── logo.png  
│  
├── utils/  
│   ├── search_engine.py  
│   ├── data_processor.py  
│   └── filters.py  
│  
└── .streamlit/config.toml  

---

## 🧠 How It Works

- TF-IDF Vectorization  
- Cosine Similarity  
- City Filters  
- Vehicle Filters  
- Duplicate Removal Logic  

---

## 📊 Database Statistics

| Metric | Count |
|--------|-------|
| Total Rules | 500 |
| Cities | 10 |
| Vehicles | 5 |
| Offense Types | 10 |

---

## 📖 Usage Guide

- Enter Query  
- Choose City  
- Choose Vehicle  
- Check Suggestions  
- Switch Theme  

---

## 🤝 Contributing

1. Fork  
2. Create Branch  
3. Commit  
4. Push  
5. Create PR  

---

## 📄 License

Distributed under MIT License.  
See LICENSE for details.

---

## 📞 Contact

Maintainer: **Tanish Khokha**  
GitHub: https://github.com/tanish152  
Project: https://github.com/tanish152/AI-Traffic-Rules-Chatbot---India  
Live Demo: https://aitrafficrulechatbot.streamlit.app/

---

## 🙏 Acknowledgments

- Streamlit  
- Scikit-learn  
- MORTH  
- Contributors  
