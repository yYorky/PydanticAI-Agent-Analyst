from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic import BaseModel
import yfinance as yf
import gradio as gr
from dotenv import load_dotenv
import os
import nest_asyncio
import pandas as pd
import math
import matplotlib.pyplot as plt
import base64

nest_asyncio.apply()
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

groq_model = GroqModel(
    model_name="llama-3.1-8b-instant",  # tokens per min at 20,000 & Context Window 128K
    api_key=groq_api_key
)

class AnalystReport(BaseModel):
    symbol: str
    executive_summary: str
    company_overview: str
    industry_and_market_analysis: str
    investment_thesis: str
    financial_analysis: str
    risks_and_concerns: str
    catalysts: str
    technical_analysis: str
    esg_analysis: str
    recommendations: str
    appendices_and_disclosures: str

def format_number(x):
    if isinstance(x, (int, float)) and not math.isnan(x):
        if abs(x) >= 1e12:
            return f"{x / 1e12:.2f}T"
        elif abs(x) >= 1e9:
            return f"{x / 1e9:.2f}B"
        elif abs(x) >= 1e6:
            return f"{x / 1e6:.2f}M"
        elif abs(x) >= 1e3:
            return f"{x / 1e3:.2f}K"
        else:
            return f"{x:,.2f}"
    return str(x)

# def format_dataframe(df: pd.DataFrame, max_rows=10):
#     if df is None or df.empty:
#         return "No data available."
#     # Apply formatting
#     formatted_df = df.copy()
#     for col in formatted_df.columns:
#         formatted_df[col] = formatted_df[col].apply(format_number)
#     return formatted_df.head(max_rows).to_markdown(index=True, tablefmt="pipe")

def format_dataframe(df: pd.DataFrame, max_rows=10):
    """
    Format a DataFrame and return it as an HTML table.
    """
    if df is None or df.empty:
        return "<p>No data available.</p>"
    
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(format_number)
    return formatted_df.head(max_rows).to_html(index=True, border=1, escape=False)

def get_sma(df, window=50):
    return df['Close'].rolling(window=window).mean()

def summarize_news(news_list):
    """
    Summarize the top news and provide links in point form.
    
    Parameters:
        news_list (list): A list of news articles where each article is a dictionary with 'title' and 'link' keys.
    
    Returns:
        str: A formatted string of news summaries with links in markdown format.
    """
    if not news_list or len(news_list) == 0:
        return "No news available."
    
    summaries = []
    for article in news_list[:5]:  # Get top 5 articles
        title = article.get('title', 'No Title')
        link = article.get('link', '#')
        summaries.append(f"- [{title}]({link})")
    
    return "\n".join(summaries)

def summarize_sector_overview(sector_overview, sector_key):
    """
    Converts the raw sector overview dictionary and sectorKey into a human-readable summary.

    Parameters:
        sector_overview (dict): Raw sector overview data.
        sector_key (str): The sector name or key.

    Returns:
        str: A formatted paragraph summarizing the sector.
    """
    if not sector_overview or "error" in sector_overview:
        return "No sector overview information available."
    
    sector_name = sector_key if sector_key != 'N/A' else "Unknown Sector"
    description = sector_overview.get('description', 'No description available.')
    companies_count = sector_overview.get('companies_count', 'N/A')
    industries_count = sector_overview.get('industries_count', 'N/A')
    market_cap = sector_overview.get('market_cap', 'N/A')
    market_weight = sector_overview.get('market_weight', 'N/A')
    employee_count = sector_overview.get('employee_count', 'N/A')

    # Format numbers for readability
    market_cap_formatted = format_number(market_cap)
    market_weight_formatted = f"{market_weight * 100:.2f}%" if isinstance(market_weight, float) else market_weight
    employee_count_formatted = format_number(employee_count)

    # Construct the summary paragraph
    summary = (
        f"The {sector_name} sector comprises {companies_count} companies across {industries_count} industries, with a total market "
        f"capitalization of {market_cap_formatted}. It represents {market_weight_formatted} of the overall market and "
        f"employs approximately {employee_count_formatted} individuals. {description}"
    )
    return summary
  
def plot_technical_analysis(data, ticker, filename="technical_analysis.png"):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(data.index, data['Close'], label='Close Price', alpha=0.8)
    plt.plot(data.index, data['SMA50'], label='50-day SMA', alpha=0.8)
    plt.plot(data.index, data['SMA200'], label='200-day SMA', alpha=0.8)
    plt.title(f"{ticker} Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return filename

def get_stock_data(symbol: str):
    stock = yf.Ticker(symbol)

    price = stock.fast_info.last_price
    info = stock.info

    # Fetch optional data:
    def safe_getattr(obj, attr):
        try:
            return getattr(obj, attr)
        except:
            return None

    recommendations = safe_getattr(stock, 'recommendations')
    financials = safe_getattr(stock, 'financials')
    quarterly_financials = safe_getattr(stock, 'quarterly_financials')
    sustainability = safe_getattr(stock, 'sustainability')
    earnings_estimate = safe_getattr(stock, 'earnings_estimate')
    revenue_estimate = safe_getattr(stock, 'revenue_estimate')
    earnings_history = safe_getattr(stock, 'earnings_history')
    insider_transactions = safe_getattr(stock, 'insider_transactions')
    hist = stock.history(period="1y")

    # Directly fetch stock news
    news = stock.news if isinstance(stock.news, list) else []
    
    # Format sector overview and top companies
    sector_key = info.get('sectorKey', 'N/A')
    if sector_key and sector_key != 'N/A':
        try:
            sector = yf.Sector(sector_key)
            sector_overview = sector.overview
            sector_top_companies = sector.top_companies[:10]
        except Exception as e:
            sector_overview = {"error": f"Could not fetch sector overview: {str(e)}"}
            sector_top_companies = pd.DataFrame()
    else:
        sector_overview = {"error": "No sector key found for this company."}
        sector_top_companies = pd.DataFrame()

    if hist.empty:
        sma_50 = pd.Series()
        sma_200 = pd.Series()
        last_close = None
    else:
        sma_50 = get_sma(hist, 50)
        sma_200 = get_sma(hist, 200)
        last_close = hist['Close'].iloc[-1]

    return {
        "price": price,
        "info": info,
        "recommendations": recommendations if isinstance(recommendations, pd.DataFrame) else pd.DataFrame(),
        "financials": financials if isinstance(financials, pd.DataFrame) else pd.DataFrame(),
        "quarterly_financials": quarterly_financials if isinstance(quarterly_financials, pd.DataFrame) else pd.DataFrame(),
        "sustainability": sustainability if isinstance(sustainability, pd.DataFrame) else pd.DataFrame(),
        "earnings_estimate": earnings_estimate if isinstance(earnings_estimate, pd.DataFrame) else pd.DataFrame(),
        "revenue_estimate": revenue_estimate if isinstance(revenue_estimate, pd.DataFrame) else pd.DataFrame(),
        "earnings_history": earnings_history if isinstance(earnings_history, pd.DataFrame) else pd.DataFrame(),
        "news": news,  # Add news directly from stock.news
        "insider_transactions": insider_transactions if isinstance(insider_transactions, pd.DataFrame) else pd.DataFrame(),
        "hist": hist,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "last_close": last_close,
        "sector_overview": sector_overview,  # Add sector overview
        "sector_top_companies": sector_top_companies,  # Add sector top companies
        "sector_key": sector_key
    }

stock_agent = Agent(
    model=groq_model,
    result_type=AnalystReport
)

def get_stock_report(query):
    symbol = query.strip().upper()
    data = get_stock_data(symbol)

    price = data["price"]
    info = data["info"]
    news = data["news"]  # Extract the news directly
    recommendations = data["recommendations"]
    financials = data["financials"]
    quarterly_financials = data["quarterly_financials"]
    sustainability = data["sustainability"]
    earnings_estimate = data["earnings_estimate"]
    revenue_estimate = data["revenue_estimate"]
    earnings_history = data["earnings_history"]
    insider_transactions = data["insider_transactions"]
    hist = data["hist"]
    sma_50 = data["sma_50"]
    sma_200 = data["sma_200"]
    last_close = data["last_close"]
    sector_overview = data["sector_overview"]
    sector_top_companies = data["sector_top_companies"]
    sector_key = data["sector_key"]  # Extract sector_key here
    
    # Summarize news
    news_summaries = summarize_news(news)

    # Format DataFrames:
    rec_table = format_dataframe(recommendations)
    fin_table = format_dataframe(financials,max_rows=50)
    qfin_table = format_dataframe(quarterly_financials,max_rows=50)
    earn_est_table = format_dataframe(earnings_estimate)
    rev_est_table = format_dataframe(revenue_estimate)
    earn_hist_table = format_dataframe(earnings_history)
    insider_table = format_dataframe(insider_transactions,max_rows=5)
    sust_table = format_dataframe(sustainability,max_rows=17)
    
    # Format top companies
    sector_overview_summary = summarize_sector_overview(sector_overview, sector_key)
    top_companies_table = format_dataframe(sector_top_companies) if not sector_top_companies.empty else "No data available."
    
    # Generate technical analysis plot
    plot_filename = plot_technical_analysis(hist, symbol)
    
    # Convert plot to Base64 for embedding
    with open(plot_filename, "rb") as plot_file:
        plot_base64 = base64.b64encode(plot_file.read()).decode()

    # Technical Analysis Section
    technical_analysis_section = (
        f"![Technical Analysis Plot](data:image/png;base64,{plot_base64})\n\n"
    )

    # Target Price Table
    target_price_table = f"""
| Metric        | Value     |
|---------------|-----------|
| Current Price | ${format_number(price)} |
| Target Low    | {format_number(info.get('targetLowPrice', 'N/A'))} |
| Target High   | {format_number(info.get('targetHighPrice', 'N/A'))} |
| Target Mean   | {format_number(info.get('targetMeanPrice', 'N/A'))} |
| Target Median | {format_number(info.get('targetMedianPrice', 'N/A'))} |
"""

    # Construct the System Prompt
    system_prompt = (
        "You are an equity research analyst tasked with generating analyst report for stock data.\n"
        "Generate concise relevant insights for the following sections:\n\n"
        "1. Executive Summary: A concise summary, include Ticker, Current Price, Recommendation considering the Current Price and the Target Price table and recommendations table, and rationale.\n"
        "2. Company Overview: Include Company Name, Sector, Industry, Website, and Business Summary.\n"
        "3. Industry and Market Analysis: Include the sector overview, trends, competitive landscape, and summarize the top companies in the sector.\n"
        "4. Investment Thesis: Summarize sentiment and use recommendations table.\n"
        "5. Financial Analysis: Summarize insights from annual and quarterly financials.\n"
        "6. Risks and Concerns: Summarize risks and controversies.\n"
        "7. Catalysts: Highlight earnings, revenue, history, and news.\n"
        "8. Technical Analysis: Discussion only on key metrics like last close and 50-day SMA and 200-day SMA.\n"
        "9. ESG Analysis: Summarize sustainability data and compare scores with peerEsgScorePerformance, peerGovernancePerformance, peerSocialPerformance, peerEnvironmentPerformance found in the Sustainability Table.\n"
        "10. Recommendations: Provide an overall investment recommendation taking into account the information available and Current Price with Target Price table.\n"
        "11. Appendices & Disclosures: Summarize insider transactions.\n\n"
        "Respond in JSON format conforming to the AnalystReport schema."
        f"Executive Summary:\nTicker: {symbol}\nCurrent Price: ${format_number(price)}\nTarget Price Table:\n{target_price_table}\n"
        f"Company Overview:\n{info.get('longName', 'N/A')}, {info.get('sector', 'N/A')}, {info.get('industry', 'N/A')}, {info.get('website', 'N/A')}\n"
        f"Business Summary: {info.get('longBusinessSummary', 'N/A')}\n\n"
        f"Recommendations Table:\n{rec_table}\n\n"
        f"Annual Financials:\n{fin_table}\n\n"
        f"Quarterly Financials:\n{qfin_table}\n\n"
        f"Sustainability Table:\n{sust_table}\n\n"
        f"Earnings Estimates:\n{earn_est_table}\n\n"
        f"Revenue Estimates:\n{rev_est_table}\n\n"
        f"Earnings History:\n{earn_hist_table}\n\n"
        f"Technical Analysis:\nLast Close: ${format_number(last_close)}, 50-day SMA: ${format_number(sma_50.dropna().iloc[-1] if not sma_50.dropna().empty else 'N/A')}, "
        f"200-day SMA: ${format_number(sma_200.dropna().iloc[-1] if not sma_200.dropna().empty else 'N/A')}\n\n"
        f"Insider Transactions:\n{insider_table}\n"
    )

    try:
        # Run the LLM to generate summaries
        result = stock_agent.run_sync(system_prompt)
        if not isinstance(result.data, AnalystReport):
            raise ValueError("Invalid response format from LLM.")

        # Construct the final report combining tables and summaries
        report = (
            f"**Symbol:** {result.data.symbol}\n\n"
            f"# Executive Summary\n{result.data.executive_summary}\n\n{target_price_table}\n\n"
            f"# Company Overview\n{result.data.company_overview}\n\n"
            f"# Industry and Market Analysis\n"
            f"### Sector Overview\n{sector_overview_summary}\n\n"
            f"### Top Companies\n{top_companies_table}\n\n"
            f"{result.data.industry_and_market_analysis}\n\n"
            f"# Investment Thesis\n{result.data.investment_thesis}\n\n{rec_table}\n\n"
            f"# Financial Analysis\n{result.data.financial_analysis}\n\n"
            f"### Annual Financials\n{fin_table}\n\n### Quarterly Financials\n{qfin_table}\n\n"
            f"# Risks and Concerns\n{result.data.risks_and_concerns}\n\n"
            f"# Catalysts\n{news_summaries}\n\n{result.data.catalysts}\n\n"
            f"### Earnings Estimate\n{earn_est_table}\n\n### Revenue Estimate\n{rev_est_table}\n\n"
            f"### Earnings History\n{earn_hist_table}\n\n"
            f"# Technical Analysis\n{result.data.technical_analysis}\n{technical_analysis_section}\n\n"
            f"# ESG Analysis\n{result.data.esg_analysis}\n\n{sust_table}\n\n"
            f"# Recommendations\n{result.data.recommendations}\n\n"
            f"# Appendices & Disclosures\n{result.data.appendices_and_disclosures}\n\n{insider_table}\n"
        )
        return report

    except Exception as e:
        return f"Error: {str(e)}"
    
def get_stock_report_with_progress(query):
  
    yield "Generating stock report..."
    symbol = query.strip().upper()
    
    # Call your function to fetch the report
    report = get_stock_report(symbol)
    yield report


with gr.Blocks(title="Pydantic AI Stock Analyst Agent") as demo:
    # gr.Markdown("""
    #             # Pydantic AI Stock Analyst Agent
    #             ##### Model: Llama3.1-8b-instant  
    #             ##### Hosted by: Groq  
    #             ##### Agents: Pydantic AI  
    #             ##### Market Data: Yahoo Finance
    #             """)
    gr.Markdown("![](https://raw.githubusercontent.com/yYorky/PydanticAI-Agent-Analyst/refs/heads/main/static/Cover%20photo.JPG)")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
                <div class="alert alert-warning">
                    Enter a Stock Symbol to generate a Comprehensive Analyst Report
                </div>
                """, container = True)
            symbol_input = gr.Textbox(label="Stock Symbol", placeholder="AAPL", lines=1)
            generate_button = gr.Button("Generate Report")
        
        
        with gr.Column(scale=5):
            gr.Markdown(
                """
                <div class="alert alert-warning">
                    <strong>Disclaimer:</strong>
                     The information contained in this demo is intended only for for informational purposes only and does not constitute financial advice.
                     It should not be disseminated or distributed to third parties without our prior written consent.
                     No liability whatsoever with respect to the use of this demo applications or its generated contents shall be accepted.
                </div>
                """, 
                label=None,
                container = True
            )
            report_output = gr.Markdown(label="Analyst Report", container = True)
            

    generate_button.click(
        fn=get_stock_report_with_progress,
        inputs=symbol_input,
        outputs=report_output,
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch()
