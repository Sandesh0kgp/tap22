from typing import Dict, List, Any, Optional
"""Visualization utilities for the Tap Bonds AI Platform."""
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Optional

def create_bond_comparison_chart(bonds: List[Dict], metric: str = 'yield') -> go.Figure:
    """
    Create a comparison chart for bonds based on the specified metric.

    Args:
        bonds: List of bond dictionaries containing data
        metric: Metric to compare (yield, coupon_rate, price, etc.)

    Returns:
        Plotly figure object
    """
    if not bonds:
        return None

    labels = []
    values = []

    for bond in bonds:
        # Get bond ISIN and company name for label
        isin = bond.get('isin', 'Unknown')
        company = bond.get('company_name', 'Unknown')
        label = f"{company} ({isin})"

        # Get metric value based on metric type
        if metric == 'yield' and 'yield' in bond:
            value = bond['yield']
        elif metric == 'coupon_rate':
            coupon_details = bond.get('coupon_details', {})
            if isinstance(coupon_details, dict):
                value = float(coupon_details.get('rate', 0))
            else:
                value = 0
        elif metric in bond:
            value = bond[metric]
        else:
            value = 0

        labels.append(label)
        values.append(value)

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=values,
            textposition='auto',
            marker_color='royalblue'
        )
    ])

    fig.update_layout(
        title=f"Bond Comparison by {metric.capitalize()}",
        xaxis_title="Bonds",
        yaxis_title=metric.capitalize(),
        height=500,
        margin=dict(l=50, r=50, t=80, b=100)
    )

    return fig

def create_cashflow_chart(cashflows: List[Dict[str, Any]], isin: str = None) -> go.Figure:
    """
    Create a chart visualizing cash flows over time.
    
    Args:
        cashflows: List of cash flow dictionaries
        isin: ISIN of the bond
        
    Returns:
        Plotly figure for cash flow visualization
    """
    # Extract data for chart
    dates = []
    principal_amounts = []
    interest_amounts = []
    
    for cf in cashflows:
        date_str = cf.get("cash_flow_date", "")
        if date_str:
            # Convert date string to datetime
            date_parts = date_str.split("-")
            if len(date_parts) == 3:
                date = datetime.datetime(int(date_parts[2]), int(date_parts[1]), int(date_parts[0]))
                dates.append(date)
                
                # Get amounts
                principal = cf.get("principal_amount", 0)
                interest = cf.get("interest_amount", 0)
                
                principal_amounts.append(principal)
                interest_amounts.append(interest)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        "Date": dates,
        "Principal": principal_amounts,
        "Interest": interest_amounts,
        "Total": [p + i for p, i in zip(principal_amounts, interest_amounts)]
    })
    
    # Sort by date
    df = df.sort_values("Date")
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add principal bars
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Principal"],
        name="Principal",
        marker_color="#1E3A8A"
    ))
    
    # Add interest bars
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Interest"],
        name="Interest",
        marker_color="#93C5FD"
    ))
    
    # Update layout
    title = f"Cash Flow Schedule"
    if isin:
        title += f" for {isin}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Amount (₹)",
        barmode="stack",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add a sum line
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Total"],
        mode="lines+markers",
        name="Total",
        line=dict(color="#FB923C", width=2)
    ))
    
    return fig

def create_maturity_distribution_chart(bonds: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart showing the distribution of bond maturities.
    
    Args:
        bonds: List of bond dictionaries
        
    Returns:
        Plotly figure for maturity distribution
    """
    # Extract maturity years
    years = []
    for bond in bonds:
        maturity_date = bond.get("maturity_date", "")
        if maturity_date:
            date_parts = maturity_date.split("-")
            if len(date_parts) == 3:
                year = int(date_parts[2])
                years.append(year)
    
    # Count bonds per year
    year_counts = {}
    for year in years:
        if year in year_counts:
            year_counts[year] += 1
        else:
            year_counts[year] = 1
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        "Year": list(year_counts.keys()),
        "Count": list(year_counts.values())
    }).sort_values("Year")
    
    # Create bar chart
    fig = px.bar(
        df,
        x="Year",
        y="Count",
        title="Bond Maturity Distribution by Year",
        labels={"Year": "Maturity Year", "Count": "Number of Bonds"},
        color="Count",
        color_continuous_scale="blues"
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=50, r=50, t=80, b=50),
        coloraxis_showscale=False
    )
    
    # Add value labels on top of bars
    fig.update_traces(
        texttemplate="%{y}",
        textposition="outside"
    )
    
    return fig

def create_company_metrics_radar_chart(company_data: Dict[str, Any]) -> go.Figure:
    """
    Create a radar chart for company financial metrics.
    
    Args:
        company_data: Dictionary with company metrics
        
    Returns:
        Plotly figure for company metrics radar chart
    """
    # Extract key metrics
    key_metrics = company_data.get("key_metrics", {})
    if isinstance(key_metrics, str):
        try:
            key_metrics = eval(key_metrics)
        except:
            key_metrics = {}
    
    # Select relevant metrics for radar chart
    relevant_metrics = [
        "current_ratio",
        "debt_to_equity",
        "interest_coverage_ratio",
        "return_on_equity",
        "net_profit_margin"
    ]
    
    # Prepare data for radar chart
    categories = []
    values = []
    
    for metric in relevant_metrics:
        if metric in key_metrics and key_metrics[metric] != "N/A":
            display_name = metric.replace("_", " ").title()
            categories.append(display_name)
            values.append(float(key_metrics[metric]))
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=company_data.get("company_name", "Company"),
        line_color="#1E3A8A"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]  # Add 20% padding
            )
        ),
        title=f"Key Financial Metrics for {company_data.get('company_name', 'Company')}",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

def create_yield_comparison_by_platform_chart(bonds: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart comparing bond yields across platforms.
    
    Args:
        bonds: List of bond dictionaries
        
    Returns:
        Plotly figure for yield comparison by platform
    """
    # Extract data
    isins = []
    companies = []
    platforms = []
    yields = []
    
    for bond in bonds:
        isin = bond.get("isin", "")
        company = bond.get("issuer", "Unknown")
        
        # Handle yield_range format like "7.5%-8.0%"
        yield_range = bond.get("yield_range", "0%-0%")
        min_yield, max_yield = yield_range.split("-")
        min_yield = float(min_yield.strip("%"))
        max_yield = float(max_yield.strip("%"))
        
        # Create a data point for each platform
        for platform in bond.get("platforms", []):
            isins.append(isin)
            companies.append(company)
            platforms.append(platform)
            
            # Add some slight variation to make platforms distinguishable
            if platform == "SMEST":
                yields.append(min_yield)
            else:
                yields.append(max_yield)
    
    # Create DataFrame
    df = pd.DataFrame({
        "ISIN": isins,
        "Company": companies,
        "Platform": platforms,
        "Yield": yields
    })
    
    # Create grouped bar chart
    fig = px.bar(
        df,
        x="Company",
        y="Yield",
        color="Platform",
        barmode="group",
        title="Bond Yields Comparison by Platform",
        labels={"Company": "Issuer", "Yield": "Yield (%)"},
        color_discrete_map={"SMEST": "#1E3A8A", "FixedIncome": "#93C5FD"},
        hover_data=["ISIN"]
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=50, r=50, t=80, b=80),
        xaxis_tickangle=-45,
        legend_title="Platform"
    )
    
    # Add value labels on top of bars
    fig.update_traces(
        texttemplate="%{y:.2f}%",
        textposition="outside"
    )
    
    return fig

import datetime
