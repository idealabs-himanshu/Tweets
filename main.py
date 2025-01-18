import os
import streamlit as st
import requests
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import google.generativeai as genai

# Set dummy OpenAI API key to bypass validation
os.environ['OPENAI_API_KEY'] = 'dummy_key_not_actually_used'

# Page configuration
st.set_page_config(
    page_title="CrewAI News Insights", 
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

def research_news(topic):
    """
    Perform in-depth research on the news topic
    """
    model = initialize_gemini_model()
    prompt = f"""Conduct a comprehensive research analysis on the topic: {topic}
    
    Research Objectives:
    - Identify key themes and underlying narratives
    - Provide context and historical background
    - Highlight potential implications
    - Maintain an objective, analytical approach
    
    Deliver a well-structured research summary."""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Research Generation Error: {e}")
        return f"Comprehensive research on {topic} could not be generated."

def generate_tweet_analysis(news_item):
    """
    Generate a professional tweet with deep insights
    """
    model = initialize_gemini_model()
    prompt = f"""Create a professional, insightful social media post based on:

News Context:
Title: {news_item['title']}
Snippet: {news_item['snippet']}

Guidelines:
- Craft a nuanced, professional insight
- Maintain objectivity
- Aim for 280-300 characters
- Provoke thoughtful reflection
- Include a subtle, relevant hashtag"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Tweet Generation Error: {e}")
        return f"Insight on {news_item['title']}: A nuanced perspective on recent developments."

def create_crew(topic, news_articles):
    """
    Create and configure the CrewAI crew
    """
    # Define Agents
    news_researcher = Agent(
        role='Senior News Researcher',
        goal='Conduct comprehensive research on current news topics',
        backstory='''An experienced investigative journalist with a PhD in Media Studies, 
        known for providing deep, nuanced analysis of complex global events. 
        Committed to uncovering the broader context behind headlines.''',
        verbose=True
    )

    tweet_strategist = Agent(
        role='Social Media Insights Strategist',
        goal='Transform complex news into concise, thought-provoking social media content',
        backstory='''A communication expert with a background in journalism and 
        digital media strategy. Specializes in distilling complex information 
        into engaging, professional social media narratives.''',
        verbose=True
    )

    # Define Tasks
    tasks = []
    for article in news_articles:
        # Research Task
        research_task = Task(
            description=f'Conduct an in-depth research analysis on: {article["title"]}',
            agent=news_researcher,
            expected_output='A comprehensive research summary providing context and insights'
        )

        # Tweet Generation Task
        tweet_task = Task(
            description=f'Generate a professional tweet for: {article["title"]}',
            agent=tweet_strategist,
            expected_output='A professional, nuanced social media insight'
        )

        tasks.extend([research_task, tweet_task])

    # Create Crew
    crew = Crew(
        agents=[news_researcher, tweet_strategist],
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )

    return crew

def add_sidebar():
    """Enhanced sidebar with detailed app information"""
    st.sidebar.title("CrewAI News Insights")
    st.sidebar.info("""
    ### Collaborative AI News Analysis

    🤖 Powered by CrewAI:
    - Intelligent News Research
    - Professional Insight Generation
    
    #### Workflow:
    1. Specialized Researcher Agent
    2. Social Media Strategist Agent
    3. Comprehensive News Insights

    #### Key Features:
    - Deep contextual analysis
    - Nuanced content generation
    - Multi-agent collaboration
    """)

def main():
    st.title("📰 CrewAI News Insights Generator")
    st.markdown("""
    ## Collaborative AI News Analysis

    Discover deep, nuanced insights through 
    our intelligent multi-agent system.
    """)
    
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
                    
                    # Generate research context and tweet
                    research_context = research_news(article['title'])
                    tweet_insight = generate_tweet_analysis(article)
                    
                    # Display insights
                    st.markdown(f"**🔬 Research Context {idx}:**")
                    st.write(research_context)
                    st.markdown(f"**💡 Insight {idx}:** *{tweet_insight}*")
                    st.divider()
            
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Run the app
if __name__ == "__main__":
    # Add sidebar
    add_sidebar()
    
    # Run main app
    main()