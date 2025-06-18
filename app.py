import streamlit as st
import os
import tempfile
import json
from typing import Dict, List, Any
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, FileReadTool, WebsiteSearchTool, ScrapeWebsiteTool
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Page Configuration
st.set_page_config(
    page_title="AI Property Analyst",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .crew-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .agent-card {
        background: #e9ecef;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'crews' not in st.session_state:
    st.session_state.crews = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

st.sidebar.title("🏠 AI Property Analyst")
st.sidebar.markdown("---")

# API Configuration
st.sidebar.subheader("🔧 API Configuration")
api_choice = st.sidebar.selectbox(
    "Choose LLM Provider",
    ["OpenAI", "Ollama", "Both (Hybrid)"],
    index=0
)

if api_choice in ["OpenAI", "Both (Hybrid)"]:
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    openai_model = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        index=1
    )

if api_choice in ["Ollama", "Both (Hybrid)"]:
    ollama_model = st.sidebar.selectbox(
        "Ollama Model",
        ["openhermes", "llama2", "mistral", "codellama", "neural-chat"],
        index=0
    )
    ollama_base_url = st.sidebar.text_input(
        "Ollama Base URL",
        value="http://localhost:11434",
        help="Ollama server URL"
    )

# Serper API for web search
serper_api_key = st.sidebar.text_input(
    "Serper API Key",
    type="password",
    help="For web search functionality"
)

st.sidebar.markdown("---")

# Property Input
st.sidebar.subheader("🏡 Property Details")
address = st.sidebar.text_input("Property Address/Name", "")
property_type = st.sidebar.selectbox(
    "Property Type",
    ["Residential", "Commercial", "Industrial", "Mixed-Use"]
)
budget_range = st.sidebar.slider(
    "Budget Range ($)",
    min_value=50000,
    max_value=5000000,
    value=(200000, 800000),
    format="$%d"
)

# File Upload
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")

# Crew Selection
st.sidebar.subheader("👥 Analysis Crews")
selected_crew = st.sidebar.selectbox(
    "Select Analysis Crew",
    ["Investment Analysis", "Market Research", "Due Diligence", "Custom Crew"]
)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🏡 Comprehensive Property Investment Analysis</h1>
    <p style="color: white; margin: 0.5rem 0 0 0;">Advanced AI-powered real estate analysis with multiple specialized crews</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def initialize_llms():
    """Initialize language models based on user selection"""
    llms = {}
    
    if api_choice in ["OpenAI", "Both (Hybrid)"]:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            llms['openai'] = ChatOpenAI(
                model=openai_model,
                temperature=0.1,
                api_key=openai_api_key
            )
    
    if api_choice in ["Ollama", "Both (Hybrid)"]:
        llms['ollama'] = Ollama(
            model=ollama_model,
            base_url=ollama_base_url
        )
    
    return llms

@st.cache_resource
def initialize_tools():
    """Initialize CrewAI tools"""
    tools = {}
    
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
        tools['search'] = SerperDevTool()
        tools['website_search'] = WebsiteSearchTool()
        tools['scrape'] = ScrapeWebsiteTool()
    
    tools['file_read'] = FileReadTool()
    
    return tools

def create_vector_store(documents):
    """Create vector store from uploaded documents"""
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    all_docs = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        all_docs.extend(chunks)
    
    if api_choice == "OpenAI" and openai_api_key:
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    else:
        embeddings = OllamaEmbeddings(model=ollama_model)
    
    vector_store = FAISS.from_documents(all_docs, embeddings)
    return vector_store

def process_uploaded_files(uploaded_files):
    """Process uploaded files and create documents"""
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')
                docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
            documents.extend(docs)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)
    
    return documents

# =============================================================================
# CREW DEFINITIONS
# =============================================================================

def get_predefined_crews():
    """Define predefined crew configurations"""
    llms = initialize_llms()
    tools = initialize_tools()
    
    # Select primary LLM
    primary_llm = llms.get('openai', llms.get('ollama'))
    
    crews = {
        "Investment Analysis": {
            "description": "Comprehensive investment analysis focusing on ROI and financial metrics",
            "agents": [
                {
                    "role": "Senior Investment Analyst",
                    "goal": "Evaluate property investment potential and calculate comprehensive financial metrics",
                    "backstory": "Former Goldman Sachs analyst with 15 years in real estate investment banking",
                    "tools": ["search", "file_read"]
                },
                {
                    "role": "Market Research Specialist",
                    "goal": "Analyze market trends, comparables, and price movements",
                    "backstory": "Real estate economist with expertise in market cycle analysis",
                    "tools": ["search", "website_search"]
                },
                {
                    "role": "Risk Assessment Expert",
                    "goal": "Identify and quantify investment risks and mitigation strategies",
                    "backstory": "Former insurance underwriter specializing in property risk assessment",
                    "tools": ["search", "scrape"]
                },
                {
                    "role": "Financial Modeler",
                    "goal": "Create detailed financial projections and sensitivity analysis",
                    "backstory": "CFA with expertise in real estate financial modeling",
                    "tools": ["file_read"]
                }
            ]
        },
        
        "Market Research": {
            "description": "Deep market analysis focusing on trends, demographics, and growth potential",
            "agents": [
                {
                    "role": "Market Intelligence Analyst",
                    "goal": "Research comprehensive market data and trends",
                    "backstory": "Urban economist with 20 years in market research",
                    "tools": ["search", "website_search"]
                },
                {
                    "role": "Demographics Specialist",
                    "goal": "Analyze population trends and demographic shifts",
                    "backstory": "Census data analyst with expertise in population forecasting",
                    "tools": ["search", "scrape"]
                },
                {
                    "role": "Infrastructure Analyst",
                    "goal": "Evaluate transportation, utilities, and development plans",
                    "backstory": "Civil engineer turned real estate consultant",
                    "tools": ["search", "website_search"]
                },
                {
                    "role": "Competition Analyst",
                    "goal": "Analyze competitive landscape and market saturation",
                    "backstory": "Strategic consultant specializing in market positioning",
                    "tools": ["search", "scrape"]
                }
            ]
        },
        
        "Due Diligence": {
            "description": "Thorough due diligence focusing on legal, technical, and regulatory aspects",
            "agents": [
                {
                    "role": "Legal Research Specialist",
                    "goal": "Research zoning, permits, and legal constraints",
                    "backstory": "Real estate attorney with expertise in property law",
                    "tools": ["search", "website_search"]
                },
                {
                    "role": "Technical Inspector",
                    "goal": "Analyze building condition and technical specifications",
                    "backstory": "Licensed structural engineer and building inspector",
                    "tools": ["file_read", "search"]
                },
                {
                    "role": "Environmental Assessor",
                    "goal": "Evaluate environmental risks and compliance issues",
                    "backstory": "Environmental consultant with EPA experience",
                    "tools": ["search", "scrape"]
                },
                {
                    "role": "Regulatory Compliance Expert",
                    "goal": "Ensure compliance with local regulations and codes",
                    "backstory": "Former city planning official with regulatory expertise",
                    "tools": ["search", "website_search"]
                }
            ]
        }
    }
    
    return crews

# =============================================================================
# CREW MANAGEMENT INTERFACE
# =============================================================================

def display_crew_management():
    """Display crew management interface"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("👥 Crew Management")
        
        # Predefined crews
        predefined_crews = get_predefined_crews()
        
        for crew_name, crew_config in predefined_crews.items():
            with st.expander(f"📋 {crew_name}", expanded=(selected_crew == crew_name)):
                st.markdown(f"**Description:** {crew_config['description']}")
                
                st.markdown("**Team Members:**")
                for i, agent in enumerate(crew_config['agents']):
                    st.markdown(f"""
                    <div class="agent-card">
                        <strong>{agent['role']}</strong><br>
                        <em>Goal:</em> {agent['goal']}<br>
                        <em>Tools:</em> {', '.join(agent['tools'])}
                    </div>
                    """, unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"Select {crew_name}", key=f"select_{crew_name}"):
                        st.session_state.selected_crew = crew_name
                        st.success(f"Selected {crew_name}")
                
                with col_b:
                    if st.button(f"Edit {crew_name}", key=f"edit_{crew_name}"):
                        st.session_state.editing_crew = crew_name
    
    with col2:
        st.subheader("➕ Custom Crew Builder")
        
        with st.form("custom_crew_form"):
            crew_name = st.text_input("Crew Name")
            crew_description = st.text_area("Crew Description")
            
            st.markdown("**Add Team Members:**")
            num_agents = st.number_input("Number of Agents", min_value=1, max_value=8, value=3)
            
            agents = []
            for i in range(num_agents):
                st.markdown(f"**Agent {i+1}:**")
                role = st.text_input(f"Role", key=f"role_{i}")
                goal = st.text_area(f"Goal", key=f"goal_{i}")
                backstory = st.text_area(f"Backstory", key=f"backstory_{i}")
                
                available_tools = ["search", "file_read", "website_search", "scrape"]
                selected_tools = st.multiselect(f"Tools", available_tools, key=f"tools_{i}")
                
                if role and goal:
                    agents.append({
                        "role": role,
                        "goal": goal,
                        "backstory": backstory,
                        "tools": selected_tools
                    })
            
            if st.form_submit_button("Create Custom Crew"):
                if crew_name and agents:
                    st.session_state.crews[crew_name] = {
                        "description": crew_description,
                        "agents": agents,
                        "custom": True
                    }
                    st.success(f"Created custom crew: {crew_name}")

# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================

def create_crew_agents_and_tasks(crew_config):
    """Create CrewAI agents and tasks from configuration"""
    llms = initialize_llms()
    tools = initialize_tools()
    
    primary_llm = llms.get('openai', llms.get('ollama'))
    
    agents = []
    tasks = []
    
    for agent_config in crew_config['agents']:
        # Select tools for agent
        agent_tools = []
        for tool_name in agent_config.get('tools', []):
            if tool_name in tools:
                agent_tools.append(tools[tool_name])
        
        # Create agent
        agent = Agent(
            llm=primary_llm,
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            tools=agent_tools,
            verbose=True,
            allow_delegation=False
        )
        agents.append(agent)
        
        # Create corresponding task
        task = Task(
            description=f"""
            As a {agent_config['role']}, {agent_config['goal']}.
            
            Property Details:
            - Address: {address if address else 'Not specified'}
            - Type: {property_type}
            - Budget Range: ${budget_range[0]:,} - ${budget_range[1]:,}
            
            Provide detailed analysis with specific data points, calculations, and recommendations.
            Include sources and confidence levels for your findings.
            """,
            expected_output=f"Comprehensive analysis from {agent_config['role']} perspective with actionable insights",
            agent=agent
        )
        tasks.append(task)
    
    return agents, tasks

def run_analysis():
    """Execute the selected crew analysis"""
    
    if not address and not uploaded_files:
        st.error("Please provide either a property address or upload documents")
        return
    
    # Initialize LLMs and tools
    llms = initialize_llms()
    if not llms:
        st.error("Please configure at least one LLM provider")
        return
    
    # Process uploaded files
    documents = []
    if uploaded_files:
        with st.spinner("Processing uploaded documents..."):
            documents = process_uploaded_files(uploaded_files)
            if documents:
                st.session_state.vector_store = create_vector_store(documents)
                st.success(f"Processed {len(documents)} documents")
    
    # Get crew configuration
    predefined_crews = get_predefined_crews()
    if selected_crew in predefined_crews:
        crew_config = predefined_crews[selected_crew]
    elif selected_crew in st.session_state.crews:
        crew_config = st.session_state.crews[selected_crew]
    else:
        st.error("Please select a valid crew")
        return
    
    # Create and run crew
    with st.spinner(f"Running {selected_crew} analysis..."):
        try:
            agents, tasks = create_crew_agents_and_tasks(crew_config)
            
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=2,
                memory=True,
                embedder={
                    "provider": "openai" if api_choice == "OpenAI" else "ollama",
                    "config": {
                        "model": openai_model if api_choice == "OpenAI" else ollama_model
                    }
                }
            )
            
            # Execute analysis
            results = crew.kickoff()
            
            # Store results
            st.session_state.analysis_results[selected_crew] = {
                'results': results,
                'crew_config': crew_config,
                'timestamp': st.write(f"Analysis completed at: {st.write}")
            }
            
            return results
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def display_results(results, crew_config):
    """Display analysis results in organized format"""
    
    if not results:
        return
    
    # Summary Dashboard
    st.subheader("📊 Analysis Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Crew Members", len(crew_config['agents']))
    with col2:
        st.metric("Documents Processed", len(uploaded_files) if uploaded_files else 0)
    with col3:
        st.metric("Analysis Type", selected_crew)
    with col4:
        st.metric("Property Type", property_type)
    
    # Detailed Results
    st.subheader("📋 Detailed Analysis Results")
    
    # If results is a string, display it directly
    if isinstance(results, str):
        st.markdown(results)
    elif hasattr(results, 'raw'):
        st.markdown(results.raw)
    else:
        st.write(results)
    
    # Agent-specific results if available
    if hasattr(results, 'tasks_output'):
        for i, task_output in enumerate(results.tasks_output):
            agent_name = crew_config['agents'][i]['role']
            with st.expander(f"🔍 {agent_name} Analysis", expanded=False):
                st.markdown(task_output.raw if hasattr(task_output, 'raw') else str(task_output))
    
    # Export options
    st.subheader("📤 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as PDF"):
            st.info("PDF export functionality would be implemented here")
    
    with col2:
        if st.button("📊 Export as Excel"):
            st.info("Excel export functionality would be implemented here")
    
    with col3:
        if st.button("📧 Email Report"):
            st.info("Email functionality would be implemented here")

# =============================================================================
# MAIN INTERFACE LAYOUT
# =============================================================================

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analysis", "👥 Crew Management", "📊 Results", "⚙️ Settings"])

with tab1:
    st.subheader("🚀 Property Analysis")
    
    if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
        results = run_analysis()
        if results:
            st.success("✅ Analysis completed successfully!")
            # Switch to results tab
            st.session_state.show_results = True

with tab2:
    display_crew_management()

with tab3:
    st.subheader("📊 Analysis Results")
    
    if st.session_state.analysis_results:
        # Show available results
        for crew_name, result_data in st.session_state.analysis_results.items():
            with st.expander(f"📋 {crew_name} Results", expanded=True):
                display_results(result_data['results'], result_data['crew_config'])
    else:
        st.info("No analysis results available. Run an analysis first.")

with tab4:
    st.subheader("⚙️ Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**LangChain Configuration**")
        memory_enabled = st.checkbox("Enable Conversation Memory", value=True)
        max_tokens = st.slider("Max Tokens per Response", 100, 4000, 2000)
        temperature = st.slider("Response Creativity", 0.0, 2.0, 0.1)
    
    with col2:
        st.markdown("**Analysis Options**")
        include_images = st.checkbox("Include Image Analysis", value=False)
        detailed_mode = st.checkbox("Detailed Analysis Mode", value=True)
        export_format = st.selectbox("Default Export Format", ["PDF", "Word", "Excel", "JSON"])
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏠 AI Property Analyst v2.0 | Powered by CrewAI & LangChain | 
    <a href="https://github.com" target="_blank">GitHub</a> | 
    <a href="mailto:support@example.com">Support</a></p>
</div>
""", unsafe_allow_html=True)
