import streamlit as st
import logging
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import Dict, List, Any, Tuple, Optional
import json
import tempfile
import traceback
import time

from data_loader import DataLoader
from agents.orchestrator import OrchestratorAgent
from utils.visualization import create_bond_comparison_chart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Tap Bonds AI Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bond_details" not in st.session_state:
    st.session_state.bond_details = None
if "cashflow_details" not in st.session_state:
    st.session_state.cashflow_details = None
if "company_insights" not in st.session_state:
    st.session_state.company_insights = None
if "data_loading_status" not in st.session_state:
    st.session_state.data_loading_status = {
        "bond": {"status": "not_started", "message": "Not loaded"},
        "cashflow": {"status": "not_started", "message": "Not loaded"},
        "company": {"status": "not_started", "message": "Not loaded"}
    }
if "last_load_attempt" not in st.session_state:
    st.session_state.last_load_attempt = 0
if "search_results" not in st.session_state:
    st.session_state.search_results = {}
if "data_loader" not in st.session_state:
    st.session_state.data_loader = DataLoader()
if "bond_parts_loaded" not in st.session_state:
    st.session_state.bond_parts_loaded = []

# Helper functions
def save_uploadedfile(uploadedfile):
    """Save uploaded file to a temporary location and return the path"""
    try:
        if uploadedfile is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                f.write(uploadedfile.getvalue())
                logger.info(f"Saved uploaded file to {f.name}")
                return f.name
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
    return None

def validate_csv_file(file_path: str, expected_columns: List[str]) -> Tuple[bool, str]:
    """Validate if CSV file has the expected format"""
    try:
        df_header = pd.read_csv(file_path, nrows=0)
        missing_columns = [col for col in expected_columns if col not in df_header.columns]
        
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        
        return True, "File validated successfully"
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def load_data(bond_files, cashflow_file, company_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Dict]]:
    """Load data from uploaded files with error handling"""
    bond_details, cashflow_details, company_insights = None, None, None
    status = {
        "bond": {"status": "not_started", "message": ""}, 
        "cashflow": {"status": "not_started", "message": ""}, 
        "company": {"status": "not_started", "message": ""}
    }
    
    try:
        # Load and concatenate multiple bond files
        if bond_files and any(bond_files):
            status["bond"]["status"] = "in_progress"
            
            bond_dfs = []
            for i, bf in enumerate(bond_files):
                if bf is None:
                    continue
                    
                bond_path = save_uploadedfile(bf)
                if bond_path:
                    try:
                        # Validate bond file format
                        expected_bond_columns = ['isin', 'company_name']
                        is_valid, validation_message = validate_csv_file(bond_path, expected_bond_columns)
                        
                        if not is_valid:
                            status["bond"]["status"] = "error"
                            status["bond"]["message"] = f"Bond file {i+1}: {validation_message}"
                            continue
                        
                        df = pd.read_csv(bond_path)
                        if df.empty:
                            status["bond"]["status"] = "error"
                            status["bond"]["message"] = f"Bond file {i+1} is empty"
                            continue
                            
                        bond_dfs.append(df)
                    except Exception as e:
                        status["bond"]["status"] = "error"
                        status["bond"]["message"] = f"Error reading bond file {i+1}: {str(e)}"
                    finally:
                        try:
                            os.unlink(bond_path)
                        except Exception:
                            pass
            
            if bond_dfs:
                try:
                    bond_details = pd.concat(bond_dfs, ignore_index=True)
                    bond_details = bond_details.drop_duplicates(subset=['isin'], keep='first')
                    status["bond"]["status"] = "success"
                    status["bond"]["message"] = f"Loaded {len(bond_details)} bonds"
                except Exception as e:
                    status["bond"]["status"] = "error"
                    status["bond"]["message"] = f"Error concatenating bond data: {str(e)}"
            elif status["bond"]["status"] != "error":
                status["bond"]["status"] = "error"
                status["bond"]["message"] = "No valid bond files processed"
        else:
            status["bond"]["status"] = "not_started"
            status["bond"]["message"] = "No bond files uploaded"
            
        # Load cashflow data if provided
        if cashflow_file:
            status["cashflow"]["status"] = "in_progress"
            
            cashflow_path = save_uploadedfile(cashflow_file)
            if cashflow_path:
                try:
                    # Validate cashflow file format
                    expected_cashflow_columns = ['isin', 'cash_flow_date', 'cash_flow_amount']
                    is_valid, validation_message = validate_csv_file(cashflow_path, expected_cashflow_columns)
                    
                    if not is_valid:
                        status["cashflow"]["status"] = "error"
                        status["cashflow"]["message"] = validation_message
                    else:
                        cashflow_details = pd.read_csv(cashflow_path)
                        if cashflow_details.empty:
                            status["cashflow"]["status"] = "error"
                            status["cashflow"]["message"] = "Cashflow file is empty"
                            cashflow_details = None
                        else:
                            status["cashflow"]["status"] = "success"
                            status["cashflow"]["message"] = f"Loaded {len(cashflow_details)} cashflow records"
                except Exception as e:
                    status["cashflow"]["status"] = "error"
                    status["cashflow"]["message"] = f"Error reading cashflow file: {str(e)}"
                finally:
                    try:
                        os.unlink(cashflow_path)
                    except Exception:
                        pass
        else:
            status["cashflow"]["status"] = "not_started"
            status["cashflow"]["message"] = "No cashflow file uploaded"
            
        # Load company data if provided
        if company_file:
            status["company"]["status"] = "in_progress"
            
            company_path = save_uploadedfile(company_file)
            if company_path:
                try:
                    # Validate company file format
                    expected_company_columns = ['company_name']
                    is_valid, validation_message = validate_csv_file(company_path, expected_company_columns)
                    
                    if not is_valid:
                        status["company"]["status"] = "error"
                        status["company"]["message"] = validation_message
                    else:
                        company_insights = pd.read_csv(company_path)
                        if company_insights.empty:
                            status["company"]["status"] = "error"
                            status["company"]["message"] = "Company file is empty"
                            company_insights = None
                        else:
                            status["company"]["status"] = "success"
                            status["company"]["message"] = f"Loaded {len(company_insights)} company records"
                except Exception as e:
                    status["company"]["status"] = "error"
                    status["company"]["message"] = f"Error reading company file: {str(e)}"
                finally:
                    try:
                        os.unlink(company_path)
                    except Exception:
                        pass
        else:
            status["company"]["status"] = "not_started"
            status["company"]["message"] = "No company file uploaded"
            
    except Exception as e:
        logger.error(f"Unexpected error during data loading: {str(e)}")
        
        # Update any status that hasn't been set yet
        for key in status:
            if status[key]["status"] in ["not_started", "in_progress"]:
                status[key]["status"] = "error"
                status[key]["message"] = "Unexpected error during processing"
    
    return bond_details, cashflow_details, company_insights, status

def get_llm(api_key, model_option, temperature, max_tokens):
    """Initialize LLM with error handling"""
    try:
        if not api_key:
            return None
        
        llm = ChatGroq(
            model=model_option,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        return None

def perform_web_search(query: str, num_results: int = 3):
    """Perform web search using DuckDuckGo"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, num_results)
        return results
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}")
        return []

# Data access functions
def get_bond_details(bond_details, isin: str = None) -> Dict:
    """Get bond details by ISIN"""
    try:
        if bond_details is None:
            return {"error": "Bond data not loaded"}
        
        if isin and isin in bond_details['isin'].values:
            row = bond_details[bond_details['isin'] == isin].iloc[0].to_dict()
            # Parse JSON fields safely
            for field in ['coupon_details', 'issuer_details', 'instrument_details']:
                try:
                    if field in row and isinstance(row[field], str):
                        row[field] = json.loads(row[field])
                except Exception:
                    row[field] = {"error": "Failed to parse JSON data"}
            return row
        return {"error": f"Bond with ISIN {isin} not found"}
    except Exception as e:
        logger.error(f"Error in get_bond_details: {str(e)}")
    return {"error": "Failed to retrieve bond details"}

def search_bond_by_text(bond_details, text: str) -> List[Dict]:
    """Search bonds by text"""
    try:
        if bond_details is None:
            return [{"error": "Bond data not loaded"}]
        
        results = []
        for _, row in bond_details.iterrows():
            if (text.lower() in str(row.get('isin', '')).lower() or 
                text.lower() in str(row.get('company_name', '')).lower()):
                results.append(row.to_dict())
        
        if not results:
            return [{"error": f"No bonds found matching '{text}'"}]
        return results
    except Exception as e:
        logger.error(f"Error in search_bond_by_text: {str(e)}")
    return [{"error": "Error searching for bonds"}]

def get_cashflow(cashflow_details, isin: str) -> List[Dict]:
    """Get cashflow schedule by ISIN"""
    try:
        if cashflow_details is None:
            return [{"error": "Cashflow data not loaded"}]
        
        if isin:
            cf_data = cashflow_details[cashflow_details['isin'] == isin]
            if not cf_data.empty:
                return cf_data.to_dict('records')
            return [{"error": f"No cashflow data for ISIN {isin}"}]
    except Exception as e:
        logger.error(f"Error in get_cashflow: {str(e)}")
    return [{"error": "Error retrieving cashflow data"}]

def search_company(company_insights, company_name: str) -> Dict:
    """Search company insights by name"""
    try:
        if company_insights is None:
            return {"error": "Company data not loaded"}
        
        if company_name:
            matches = company_insights[company_insights['company_name'].str.contains(
                company_name, case=False, na=False)]
            if not matches.empty:
                company_data = matches.iloc[0].to_dict()
                # Parse JSON fields safely
                for field in ['key_metrics', 'income_statement', 'balance_sheet', 'cashflow']:
                    try:
                        if field in company_data and isinstance(company_data[field], str):
                            company_data[field] = json.loads(company_data[field])
                    except Exception:
                        company_data[field] = {}
                return company_data
            return {"error": f"No company found matching '{company_name}'"}
    except Exception as e:
        logger.error(f"Error in search_company: {str(e)}")
    return {"error": "Error searching for company"}

def calculate_yield(bond_details, isin: str, price: float = None) -> Dict:
    """Calculate bond yield at a given price"""
    try:
        bond_data = get_bond_details(bond_details, isin)
        if "error" in bond_data:
            return bond_data
        
        if price is None:
            return {"bond": bond_data, "error": "No price provided for yield calculation"}
        
        # Get coupon rate from coupon_details
        coupon_rate = 0
        if isinstance(bond_data.get('coupon_details'), dict):
            coupon_rate = float(bond_data['coupon_details'].get('rate', 0))
        
        # Simple yield calculation (coupon/price)
        simple_yield = (coupon_rate / price) * 100
        return {
            "bond": bond_data,
            "price": price,
            "yield": round(simple_yield, 2)
        }
    except Exception as e:
        logger.error(f"Error in calculate_yield: {str(e)}")
    return {"error": "Error processing yield calculation"}

def process_query(query: str, bond_details, cashflow_details, company_insights) -> Dict:
    """Process a natural language query"""
    try:
        # Extract ISIN if present
        isin = None
        for word in query.split():
            clean_word = ''.join(c for c in word if c.isalnum() or c in ".")
            if clean_word.upper().startswith("INE") and len(clean_word) >= 10:
                isin = clean_word.upper()
                break
        
        # Extract price if present
        price = None
        for word in query.split():
            if word.startswith("$"):
                try:
                    price = float(word[1:])
                except ValueError:
                    pass
        
        # Extract company name
        company_name = None
        if "company" in query.lower():
            parts = query.lower().split("company")
            if len(parts) > 1 and len(parts[1].strip()) > 0:
                company_name = parts[1].strip()
        
        # Determine query type
        query_type = "unknown"
        query_lower = query.lower()
        if "cash flow" in query_lower or "cashflow" in query_lower:
            query_type = "cashflow"
        elif "yield" in query_lower or "calculate" in query_lower:
            query_type = "yield"
        elif "company" in query_lower or "issuer" in query_lower:
            query_type = "company"
        elif "detail" in query_lower or "information" in query_lower or "about" in query_lower:
            query_type = "bond"
        elif "search" in query_lower or "find" in query_lower or "web" in query_lower:
            query_type = "web_search"
        
        # Prepare context
        context = {
            "query": query,
            "query_type": query_type,
            "isin": isin,
            "company_name": company_name,
            "price": price
        }
        
        # Check for data availability
        if bond_details is None and query_type in ["bond", "yield"]:
            context["data_status"] = "Bond data not loaded. Please upload bond files."
        if cashflow_details is None and query_type == "cashflow":
            context["data_status"] = "Cashflow data not loaded. Please upload cashflow file."
        if company_insights is None and query_type == "company":
            context["data_status"] = "Company data not loaded. Please upload company file."
        
        # Only fetch data if we have the necessary files loaded
        if "data_status" not in context:
            if query_type == "web_search":
                search_term = query.replace("search", "").replace("find", "").replace("web", "").strip()
                context["web_search"] = perform_web_search(search_term)
            elif query_type == "bond" and isin:
                context["bond_data"] = get_bond_details(bond_details, isin)
            elif query_type == "bond" and company_name:
                context["search_results"] = search_bond_by_text(bond_details, company_name)
            elif query_type == "cashflow" and isin:
                context["cashflow_data"] = get_cashflow(cashflow_details, isin)
                context["bond_data"] = get_bond_details(bond_details, isin)
            elif query_type == "company" and company_name:
                context["company_data"] = search_company(company_insights, company_name)
            elif query_type == "yield" and isin:
                context["yield_data"] = calculate_yield(bond_details, isin, price)
            else:
                # Try to find relevant info without specific query type
                if isin:
                    context["bond_data"] = get_bond_details(bond_details, isin)
                    if cashflow_details is not None:
                        context["cashflow_data"] = get_cashflow(cashflow_details, isin)
                if company_name and company_insights is not None:
                    context["company_data"] = search_company(company_insights, company_name)
        
        return context
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"error": f"Error processing query: {str(e)}"}

def generate_response(context: Dict, llm) -> str:
    """Generate a response using LLM based on context"""
    try:
        if llm is None:
            return "Please enter a valid GROQ API key in the sidebar to continue."
        
        if "error" in context:
            return f"Error: {context['error']}"
            
        if "data_status" in context:
            return f"I need more data to answer your question. {context['data_status']}"
        
        template = """You are a helpful financial assistant specializing in bonds.
        User Query: {query}
        Query Type: {query_type}
        
        Available Context:
        {context_str}
        
        Respond in a professional, friendly manner with Markdown formatting.
        If you cannot answer from the provided data, politely say so.
        """
        
        # Convert context to a formatted string
        context_parts = []
        for key, value in context.items():
            if key not in ["query", "query_type"] and value:
                context_parts.append(f"{key}: {value}")
        
        context_str = "\n".join(context_parts)
        
        # Create and run the chain
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": context["query"], "query_type": context["query_type"], "context_str": context_str})
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

def display_status_indicator(status):
    """Display status indicator with icon"""
    if status == "success":
        return "✅"
    elif status == "error":
        return "❌"
    elif status == "in_progress":
        return "⏳"
    else:
        return "⚪"  # not started

# Apply custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.75rem;
    }
    .financial-text {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .info-box {
        background-color: #F0F7FF;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .agent-name {
        color: #1E40AF;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .highlight-text {
        color: #1E40AF;
        font-weight: 600;
    }
    .secondary-text {
        color: #6B7280;
        font-size: 0.9rem;
    }
    .key-value {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .icon-text {
        display: flex;
        align-items: center;
    }
    .navbar {
        background-color: #1E3A8A;
        padding: 1rem;
        color: white;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .example-query {
        border: 1px dashed #CBD5E1;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    .example-query:hover {
        background-color: #F1F5F9;
    }
    </style>
    """, unsafe_allow_html=True)

# Display functions
def display_bond_details(bond_details):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="agent-name">{bond_details.get("agent_name", "Bond Directory")}</div>', unsafe_allow_html=True)
    
    bond = bond_details.get("bond", {})
    
    if not bond or (isinstance(bond, dict) and "error" in bond):
        st.markdown(f'<div class="highlight-text">Error: {bond.get("error", "Bond details not found")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown(f'<div class="highlight-text">Bond Details: {bond.get("isin", "")}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="financial-text"><strong>Issuer:</strong> {bond.get("company_name", "")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="financial-text"><strong>Issue Date:</strong> {bond.get("issue_date", "")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="financial-text"><strong>Maturity Date:</strong> {bond.get("maturity_date", "")}</div>', unsafe_allow_html=True)
    
    with col2:
        coupon_details = bond.get("coupon_details", {})
        if isinstance(coupon_details, str) and coupon_details.startswith("{"):
            try:
                coupon_details = json.loads(coupon_details)
            except:
                coupon_details = {}
                
        coupon_rate = coupon_details.get("coupon_rate", "") if isinstance(coupon_details, dict) else ""
        st.markdown(f'<div class="financial-text"><strong>Coupon Rate:</strong> {coupon_rate}</div>', unsafe_allow_html=True)
        
        frequency = coupon_details.get("frequency", "") if isinstance(coupon_details, dict) else ""
        st.markdown(f'<div class="financial-text"><strong>Payment Frequency:</strong> {frequency}</div>', unsafe_allow_html=True)
        
        bond_type = bond.get("bond_type", "")
        st.markdown(f'<div class="financial-text"><strong>Bond Type:</strong> {bond_type}</div>', unsafe_allow_html=True)
    
    if "text" in bond_details:
        st.markdown(f'<div class="financial-text">{bond_details["text"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_bonds_list(bonds_list):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="agent-name">{bonds_list.get("agent_name", "Bond Directory")}</div>', unsafe_allow_html=True)
    
    if "text" in bonds_list:
        st.markdown(f'<div class="financial-text">{bonds_list["text"]}</div>', unsafe_allow_html=True)
    
    bonds = bonds_list.get("bonds", [])
    
    if not bonds:
        st.markdown('<div class="financial-text">No bonds found matching your query.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Create a DataFrame from bonds for better display
    bonds_df = pd.DataFrame(bonds)
    
    # Select columns to display
    display_cols = ["isin", "company_name", "issue_date", "maturity_date", "bond_type"]
    display_cols = [col for col in display_cols if col in bonds_df.columns]
    
    if display_cols:
        st.dataframe(bonds_df[display_cols], use_container_width=True)
    else:
        st.dataframe(bonds_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_cashflows(cashflows):
    st.markdown('<div class="highlight-text">Cash Flow Schedule</div>', unsafe_allow_html=True)
    
    isin = cashflows.get("isin", "")
    flow_data = cashflows.get("flows", [])
    
    if isin and flow_data:
        st.markdown(f'<div class="financial-text">Cash flow schedule for ISIN: {isin}</div>', unsafe_allow_html=True)
        
        # Clean up NaN values in cash flow data
        cleaned_flows = []
        for flow in flow_data:
            cleaned_flow = {}
            for key, value in flow.items():
                # Replace NaN, None, or placeholder values with empty strings for display purposes
                if pd.isna(value) or value is None or (isinstance(value, str) and value.strip() == "#######"):
                    cleaned_flow[key] = ""
                else:
                    cleaned_flow[key] = value
            cleaned_flows.append(cleaned_flow)
        
        df = pd.DataFrame(cleaned_flows)
        
        # Format display columns
        display_cols = ["cash_flow_date", "cash_flow_amount", "principal_amount", "interest_amount", "remaining_principal", "state"]
        display_cols = [col for col in display_cols if col in df.columns]
        
        if not display_cols:
            display_cols = df.columns
        
        # Fill NaN values with empty strings for display
        display_df = df[display_cols].fillna("")
        st.dataframe(display_df, use_container_width=True)
        
        # Prepare a clean dataframe with NaN converted to 0 for calculations
        calc_df = df.copy()
        numeric_cols = ["cash_flow_amount", "principal_amount", "interest_amount", "remaining_principal"]
        for col in numeric_cols:
            if col in calc_df.columns:
                calc_df[col] = pd.to_numeric(calc_df[col], errors='coerce').fillna(0)
        
        # Calculate and display summary statistics (only if dataframe is not empty)
        if not calc_df.empty and "cash_flow_amount" in calc_df.columns:
            total_cashflow = calc_df["cash_flow_amount"].sum()
            st.markdown(f'<div class="key-value"><span>Total Cash Flow:</span> <span class="highlight-text">₹{total_cashflow:,.2f}</span></div>', unsafe_allow_html=True)
        
        if not calc_df.empty and "principal_amount" in calc_df.columns and "interest_amount" in calc_df.columns:
            total_principal = calc_df["principal_amount"].sum()
            total_interest = calc_df["interest_amount"].sum()
            st.markdown(f'<div class="key-value"><span>Total Principal:</span> <span>₹{total_principal:,.2f}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="key-value"><span>Total Interest:</span> <span>₹{total_interest:,.2f}</span></div>', unsafe_allow_html=True)

def display_company_analysis(company_analysis):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="agent-name">{company_analysis.get("agent_name", "Company Analysis")}</div>', unsafe_allow_html=True)
    
    if "text" in company_analysis:
        st.markdown(f'<div class="financial-text">{company_analysis["text"]}</div>', unsafe_allow_html=True)
    
    company = company_analysis.get("company", {})
    
    if not company or (isinstance(company, dict) and "error" in company):
        st.markdown(f'<div class="highlight-text">Error: {company.get("error", "Company data not found")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Company overview
    st.markdown(f'<div class="highlight-text">Company Overview: {company.get("company_name", "")}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="financial-text"><strong>Industry:</strong> {company.get("industry", "")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="financial-text"><strong>Sector:</strong> {company.get("sector", "")}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="financial-text"><strong>Exchange:</strong> {company.get("exchange", "")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="financial-text"><strong>Listed:</strong> {company.get("listed", "")}</div>', unsafe_allow_html=True)
    
    # Company metrics section
    metrics = company_analysis.get("metrics", {})
    if metrics:
        st.markdown('<div class="highlight-text">Key Financial Metrics</div>', unsafe_allow_html=True)
        
        # Helper function to display financial data
        def display_financial_data(data_dict):
            # Ensure the data is a dictionary
            if not isinstance(data_dict, dict):
                return
            
            for key, value in data_dict.items():
                # Skip technical fields or null values
                if key in ["__type", "_type"] or value is None:
                    continue
                
                # Format the key for display
                display_key = key.replace("_", " ").title()
                
                # Format numbers
                if isinstance(value, (int, float)):
                    if value > 1000000:
                        value = f"₹{value/1000000:.2f}M"
                    elif value > 1000:
                        value = f"₹{value/1000:.2f}K"
                    else:
                        value = f"₹{value:.2f}"
                
                st.markdown(f'<div class="key-value"><span>{display_key}:</span> <span>{value}</span></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get metrics from different sources, depending on the data structure
            metrics_data = metrics
            if isinstance(company.get("key_metrics"), dict) and not metrics:
                metrics_data = company.get("key_metrics", {})
            
            # Display the first half of metrics
            display_financial_data(dict(list(metrics_data.items())[:len(metrics_data)//2]))
        
        with col2:
            # Display the second half of metrics
            display_financial_data(dict(list(metrics_data.items())[len(metrics_data)//2:]))
    
    # Chart section
    if "chart" in company_analysis and company_analysis["chart"] is not None:
        st.plotly_chart(company_analysis["chart"], use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Display example queries
def display_example_queries():
    st.markdown("### Example Queries")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="example-query" onclick="document.querySelector(\'[data-testid=stTextInput] input\').value=\'Show details for bond INE08XP07258\'; document.querySelector(\'[data-testid=stTextInput] input\').dispatchEvent(new Event(\'input\', { bubbles: true }));">Show details for bond INE08XP07258</div>', unsafe_allow_html=True)
        st.markdown('<div class="example-query" onclick="document.querySelector(\'[data-testid=stTextInput] input\').value=\'What bonds are issued by Tata Motors?\'; document.querySelector(\'[data-testid=stTextInput] input\').dispatchEvent(new Event(\'input\', { bubbles: true }));">What bonds are issued by Tata Motors?</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="example-query" onclick="document.querySelector(\'[data-testid=stTextInput] input\').value=\'What is the cash flow schedule for INE126M07017?\'; document.querySelector(\'[data-testid=stTextInput] input\').dispatchEvent(new Event(\'input\', { bubbles: true }));">What is the cash flow schedule for INE126M07017?</div>', unsafe_allow_html=True)
        st.markdown('<div class="example-query" onclick="document.querySelector(\'[data-testid=stTextInput] input\').value=\'Which bonds are maturing in 2025?\'; document.querySelector(\'[data-testid=stTextInput] input\').dispatchEvent(new Event(\'input\', { bubbles: true }));">Which bonds are maturing in 2025?</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    try:
        # Apply CSS
        apply_custom_css()
        
        # Sidebar for configuration
        with st.sidebar:
            st.title("Configuration")
            # Try to get API key from secrets first, otherwise ask user
            api_key = st.secrets["groq"]["api_key"] if "groq" in st.secrets else ""
            api_key = st.text_input("Enter your GROQ API Key", value=api_key, type="password")
            
            st.markdown("### Upload Data Files")
            # Multiple bond files upload (max 2 for simplicity)
            st.markdown("#### Bond Detail Files")
            bond_files = []
            for i in range(2):
                bond_file = st.file_uploader(f"Upload Bond Details CSV Part {i+1}", type=["csv"], key=f"bond_file_{i}")
                bond_files.append(bond_file)
            
            cashflow_file = st.file_uploader("Upload Cashflow Details CSV", type=["csv"])
            company_file = st.file_uploader("Upload Company Insights CSV", type=["csv"])
            
            model_option = st.selectbox(
                "Select Model",
                ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
            )
            
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            max_tokens = st.slider("Max Tokens", min_value=500, max_value=4000, value=1500, step=500)
            
            if st.button("Load Data"):
                with st.spinner("Loading data..."):
                    st.session_state.last_load_attempt = time.time()
                    st.session_state.bond_details, st.session_state.cashflow_details, st.session_state.company_insights, st.session_state.data_loading_status = load_data(
                        bond_files, cashflow_file, company_file
                    )
                    if (st.session_state.bond_details is not None or 
                        st.session_state.cashflow_details is not None or 
                        st.session_state.company_insights is not None):
                        st.success("Data loaded successfully!")
                    else:
                        st.error("Failed to load data. Check the Debug Information.")
            
            # Also support the legacy approach with bond parts
            st.markdown("---")
            st.markdown("### Legacy Data Loading")
            bond_part = st.file_uploader("Upload Bond Details CSV (Part)", type=["csv"], key="legacy_bond_part")
            
            if st.button("Load Bond Details Part"):
                if bond_part is not None:
                    with st.spinner("Loading bond data..."):
                        bond_filename = f"temp_bond_part_{len(st.session_state.bond_parts_loaded)}.csv"
                        
                        with open(bond_filename, "wb") as f:
                            f.write(bond_part.getvalue())
                        
                        if st.session_state.data_loader.load_bond_details(bond_filename) is not None:
                            st.session_state.bond_parts_loaded.append(bond_filename)
                            st.success(f"✅ Loaded bond part {len(st.session_state.bond_parts_loaded)}")
                        else:
                            st.error("Failed to load bond data")
                else:
                    st.error("Please upload a file first")
            
            # Debug section
            with st.expander("Debug Info"):
                st.write("Bond Parts Loaded:", st.session_state.bond_parts_loaded)
                if st.button("Clear All Data"):
                    st.session_state.data_loader = DataLoader()
                    st.session_state.bond_parts_loaded = []
                    st.session_state.chat_history = []
                    st.session_state.bond_details = None
                    st.session_state.cashflow_details = None
                    st.session_state.company_insights = None
                    st.session_state.data_loading_status = {
                        "bond": {"status": "not_started", "message": "Not loaded"},
                        "cashflow": {"status": "not_started", "message": "Not loaded"},
                        "company": {"status": "not_started", "message": "Not loaded"}
                    }
                    st.rerun()
        
        # Main content
        st.title("Tap Bonds AI Platform")
        st.markdown("""
        Welcome to the Tap Bonds AI Platform! 💼🔍
        
        Ask about bonds, companies, cash flows, yields, or search web for more information.
        
        **Example queries:**  
        - "Show details for INE08XP07258"  
        - "What's the cash flow schedule for INE08XP07258?"  
        - "Calculate yield for INE08XP07258 at $96.50"
        - "Search web for recent Indian bond market trends"
        """)
        
        # Data status section
        st.markdown("### Data Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bond_status = st.session_state.data_loading_status.get("bond") or {"status": "not_started", "message": "Not loaded"}
            st.markdown(f"{display_status_indicator(bond_status['status'])} **Bond Data:** {bond_status['message']}")
            
        with col2:
            cashflow_status = st.session_state.data_loading_status.get("cashflow") or {"status": "not_started", "message": "Not loaded"}
            st.markdown(f"{display_status_indicator(cashflow_status['status'])} **Cashflow Data:** {cashflow_status['message']}")
            
        with col3:
            company_status = st.session_state.data_loading_status.get("company") or {"status": "not_started", "message": "Not loaded"}
            st.markdown(f"{display_status_indicator(company_status['status'])} **Company Data:** {company_status['message']}")
        
        # Legacy data status
        if st.session_state.bond_parts_loaded:
            st.info(f"Legacy bond parts loaded: {len(st.session_state.bond_parts_loaded)}")
        
        # Check for API key
        if not api_key:
            st.warning("⚠️ Please enter your GROQ API key in the sidebar to interact with the chatbot.")
        
        st.markdown("---")
        
        # Create columns for input
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("Enter your query:", key="query_input")
        with col2:
            submit_button = st.button("Submit", use_container_width=True)
        
        # Display example queries section
        display_example_queries()
        
        # Process query when Submit is clicked
        if submit_button and query:
            # Initialize LLM only when needed
            llm = get_llm(api_key, model_option, temperature, max_tokens)
            
            with st.spinner("Processing your query..."):
                # Add user query to history
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                # Process query
                if not api_key:
                    response = "Please enter your GROQ API key in the sidebar to continue."
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    # Try both options - new efficient approach and the classic orchestrator
                    try:
                        # First try the new approach with Groq
                        context = process_query(
                            query, 
                            st.session_state.bond_details,
                            st.session_state.cashflow_details,
                            st.session_state.company_insights
                        )
                        
                        response = generate_response(context, llm)
                        
                        # Add bot response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        # Fallback to the orchestrator if the new approach fails
                        try:
                            # Create orchestrator and process query with the old way
                            orchestrator = OrchestratorAgent(st.session_state.data_loader)
                            response = orchestrator.process_query(query)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        except Exception as e2:
                            # Both approaches failed
                            error_msg = f"Error processing query: {str(e)}. Fallback also failed: {str(e2)}"
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        # Display chat history
        st.markdown("### Conversation")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                content_type = message.get("content_type", "text")
                
                # Display the message differently based on content type
                if isinstance(message.get("content"), dict) and content_type == "bond_details":
                    display_bond_details(message["content"])
                elif isinstance(message.get("content"), dict) and content_type == "bonds_list":
                    display_bonds_list(message["content"])
                elif isinstance(message.get("content"), dict) and content_type == "bond_comparison":
                    display_bond_comparison(message["content"])
                elif isinstance(message.get("content"), dict) and content_type == "company_analysis":
                    display_company_analysis(message["content"])
                elif isinstance(message.get("content"), dict) and content_type == "error":
                    st.error(message["content"].get("text", "An error occurred."))
                else:
                    # Default text display
                    st.markdown(f"**Tap Bonds AI**: {message['content']}")
        
        # Add footer
        st.markdown("---")
        st.markdown("Powered by Tap Bonds AI")
        
        # Display debugging information in an expander
        with st.expander("Debug Information", expanded=False):
            st.write("Data Availability:")
            st.write(f"- Bond Details: {display_status_indicator(st.session_state.data_loading_status['bond']['status'])} {st.session_state.data_loading_status['bond']['message']}")
            st.write(f"- Cashflow Details: {display_status_indicator(st.session_state.data_loading_status['cashflow']['status'])} {st.session_state.data_loading_status['cashflow']['message']}")
            st.write(f"- Company Insights: {display_status_indicator(st.session_state.data_loading_status['company']['status'])} {st.session_state.data_loading_status['company']['message']}")
            
            # Add data sample viewer
            if st.checkbox("Show Data Samples"):
                if st.session_state.bond_details is not None:
                    st.subheader("Bond Details Sample")
                    st.dataframe(st.session_state.bond_details.head(3))
                
                if st.session_state.cashflow_details is not None:
                    st.subheader("Cashflow Details Sample")
                    st.dataframe(st.session_state.cashflow_details.head(3))
                
                if st.session_state.company_insights is not None:
                    st.subheader("Company Insights Sample")
                    st.dataframe(st.session_state.company_insights.head(3))
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Uncaught exception in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
