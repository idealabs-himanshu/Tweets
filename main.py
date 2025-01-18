import os
import streamlit as st
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="AI News Insights", 
    page_icon="📰", 
    layout="centered"
)

# Load environment variables
load_dotenv()

# API Keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def fetch_news(topic: str):
    """
    Fetch latest news using Serper API
    """
    url = "https://google.serper.dev/news"
    payload = json.dumps({
        "q": topic,
        "num": 5
    })
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        news_data = response.json()
        
        # Extract relevant news information
        news_results = []
        for item in news_data.get('news', []):
            news_results.append({
                'title': item.get('title', ''),
                'snippet': item.get('snippet', ''),
                'link': item.get('link', '')
            })
        
        return news_results
    
    except requests.RequestException as e:
        st.error(f"News Fetch Error: {e}")
        return []

def initialize_gemini_model():
    """
    Initialize and return the Gemini model
    """
    return genai.GenerativeModel('gemini-pro')

class NewsAgent:
    def __init__(self, role, goal, backstory):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.model = initialize_gemini_model()

    def research_news(self, topic):
        """
        Perform in-depth research on the news topic
        """
        prompt = f"""As a {self.role} with the goal of {self.goal}, 
        conduct a comprehensive research analysis on the topic: {topic}
        
        Research Objectives:
        - Identify key themes and underlying narratives
        - Provide context and historical background
        - Highlight potential implications
        - Maintain an objective, analytical approach
        
        Deliver a well-structured research summary."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"Research Generation Error: {e}")
            return f"Comprehensive research on {topic} could not be generated."

    def generate_insight(self, news_item, research_context):
        """
        Generate a professional insight with deep analysis
        """
        prompt = f"""As a {self.role}, create a professional, insightful analysis based on:

News Context:
Title: {news_item['title']}
Snippet: {news_item['snippet']}

Previous Research:
{research_context}

Guidelines:
- Craft a nuanced, professional insight
- Integrate research context
- Maintain objectivity
- Aim for 300-350 characters
- Provoke thoughtful reflection
- Include a subtle, relevant perspective"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"Insight Generation Error: {e}")
            return f"Insight on {news_item['title']}: A nuanced perspective on recent developments."

def add_sidebar():
    """Enhanced sidebar with detailed app information"""
    st.sidebar.title("AI News Insights")
    st.sidebar.info("""
    ### Intelligent News Analysis

    🧠 Powered by:
    - Serper News API
    - Google Gemini AI
    
    #### Features:
    - Deep news research
    - Contextual insights
    - Objective analysis

    #### How It Works:
    1. Enter a news topic
    2. Retrieve latest news
    3. Generate in-depth insights
    """)

def main():
    st.title("📰 AI-Powered News Insights")
    st.markdown("""
    ## Discover Deeper Perspectives

    Transform headlines into comprehensive insights 
    with our AI-driven analysis tool.
    """)
    
    # Initialize Agents
    researcher = NewsAgent(
        role='Senior News Analyst',
        goal='Provide comprehensive and nuanced news analysis',
        backstory='An experienced investigative journalist with expertise in global affairs and deep contextual understanding.'
    )
    
    # Topic input
    topic = st.text_input(
        "Enter a news topic", 
        placeholder="E.g., Global Technology, Political Developments, Economic Trends"
    )
    
    # Generate button
    if st.button("Generate Insights", type="primary"):
        if not topic:
            st.warning("Please enter a news topic.")
            return
        
        # Fetch and process news
        with st.spinner("Searching for relevant news..."):
            try:
                # Fetch news articles
                news_articles = fetch_news(topic)
                
                if not news_articles:
                    st.warning("No news found. Try a different topic.")
                    return
                
                # Generate and display insights
                st.subheader("🔍 News Insights")
                
                for idx, article in enumerate(news_articles, 1):
                    # Display news article details
                    st.markdown(f"""
                    **Article {idx}:**
                    - **Title:** {article['title']}
                    - **Snippet:** {article['snippet']}
                    - **Source Link:** {article['link']}
                    """)
                    
                    # Generate research context and insight
                    research_context = researcher.research_news(article['title'])
                    insight = researcher.generate_insight(article, research_context)
                    
                    # Display insights
                    st.markdown(f"**🔬 Research Context {idx}:**")
                    st.write(research_context)
                    st.markdown(f"**💡 Insight {idx}:** *{insight}*")
                    st.divider()
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Run the app
if __name__ == "__main__":
    # Add sidebar
    add_sidebar()
    
    # Run main app
    main()
