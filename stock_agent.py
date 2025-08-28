import yfinance as yf
import pandas as pd
import numpy as np
import json
import re

from typing import Dict, Annotated
from typing_extensions import TypedDict
import json
import matplotlib.pyplot as plt
import operator
from curl_cffi import requests

#from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, START, END
from langchain.prompts import PromptTemplate 
from langchain_ollama import OllamaLLM

# Define a reducer to merge dictionaries
def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge two dictionaries, combining key-value pairs."""
    result = left.copy()  # Copy left to avoid mutating
    result.update(right)  # Update with right's key-value pairs
    return result

# Initialize the local Ollama endpoint
def setup_llm():
    return OllamaLLM(
        model="deepseek-r1:14b",
        base_url="http://localhost:11434",
        temperature=0
    )

class InputState(TypedDict):
    symbol: str

class OutputState(TypedDict):
    symbol: str
    results: Annotated[Dict, merge_dicts]

class OverallState(InputState, OutputState):
    pass

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Technical Analysis Node
def technical_analysis(state: InputState) -> Dict:
    """Node for technical analysis"""
    symbol = state["symbol"]

    print ("-->Starting Technical analysis.")

    session = requests.Session(impersonate="chrome")
    # stock = yf.Ticker(symbol, session=session)
    stock = yf.Ticker(symbol)

    hist = stock.history(period='1y')
    info = stock.info

    # Calculate all indicators from YahooFinance API data
    sma_20 = hist['Close'].rolling(window=20).mean()
    sma_50 = hist['Close'].rolling(window=50).mean()
    sma_200 = hist['Close'].rolling(window=200).mean()
    rsi = calculate_rsi(hist)
    MACD, Signal = calculate_macd(hist) 
    latestMACD = MACD.iloc[-1]
    latestSignal = Signal.iloc[-1]

    data = {
        'current_price': hist['Close'].iloc[-1],
        'sma_20': sma_20.iloc[-1],
        'sma_50': sma_50.iloc[-1],
        'sma_200': sma_200.iloc[-1],
        'rsi': rsi.iloc[-1],
        'volume_trend': hist['Volume'].iloc[-5:].mean() / hist['Volume'].iloc[-20:].mean(),
        'dividendYield': info.get('dividendYield', 'N/A'),
        'beta': info.get('beta', 'N/A'),
        'PE': info.get('trailingPE', 'N/A'),
        'MACD': latestMACD,
        'Signal': latestSignal
    }
    
    print(f"Indicator Values for {symbol} as fetched from Yahoo Finance and calculated:")
    for indicator in data:
        print ("- ", indicator, ":", data[indicator])
    print ("\n")

    prompt = PromptTemplate.from_template(
        """You are an economic specialist, presenting your conclusions to a group with moderate understanding on the topic of stocks.
        Please analyze the following technical indicators for {symbol} and concisely write down your thoughts together with the reasoning behind it:
        {data}

        Please include the following topics (at least):
        1. Trend analysis
        2. Support/Resistance levels
        3. Technical rating (Bullish/Neutral/Bearish)
        4. Key signals
        """
    )

    input_data = {
        'symbol': symbol,
        'data': json.dumps(data, indent=2)
    }

    chain =  prompt | LLM_instance
    analysis = chain.invoke(input=input_data)

    if remove_think_tags:
        analysis = re.sub(r"<think>.*?</think>", "", analysis, flags = re.DOTALL)

    answer = {
        "symbol": symbol,
        "results": {
            "technical": {
                "data": data,
                "analysis": analysis
            }
        }
    }
    print ("---Technical analysis: DONE")
    if DEBUG: 
        print ("---End of TECHNICAL: \n", answer, "\n\n")
    return answer

# Market Analysis Node
def market_analysis(state: InputState) -> Dict:
    """Node for market analysis"""
    symbol = state["symbol"]

    print ("-->Starting Market analysis.")

    # Fetch market data
    session = requests.Session(impersonate="chrome")
    # stock = yf.Ticker(symbol, session=session)
    stock = yf.Ticker(symbol)
    info = stock.info

    data = {
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'market_cap': info.get('marketCap', 0),
        'beta': info.get('beta', 1.0),
        'pe_ratio': info.get('trailingPE', 0)
    }

    prompt = PromptTemplate.from_template(
        """You are a business specialist, needing to analyze the situation for stock symbol {symbol}. 
        Please analyze the market context in a concise fashion and targeted at a group with moderate understanding of business:
        {data}

        Please include:
        1. Market sentiment
        2. Sector analysis
        3. Risk assessment
        4. Market outlook
        """
    )

    input_data = {
        'symbol': symbol, 
        'data': json.dumps(data, indent=2)
    }

    chain =  prompt | LLM_instance
    analysis = chain.invoke(input=input_data)

    answer = {
        "results": {
            "market": {
                "data": data,
                "analysis": analysis
            }
        }
    }

    if remove_think_tags:
        analysis = re.sub(r"<think>.*?</think>", "", analysis, flags = re.DOTALL)

    answer = {
        "results": {
            "market": {
                "data": data,
                "analysis": analysis
            }
        }
    }
    print ("---Market analysis: DONE")
    if DEBUG: 
        print ("---End of MARKET: \n", answer, "\n\n")
    return answer

# News Analysis Node
def news_analysis(state: InputState) -> Dict:
    """Node for news analysis"""
    symbol = state["symbol"]

    print ("-->Starting News analysis.")

    # Fetch news
    session = requests.Session(impersonate="chrome")
    # stock = yf.Ticker(symbol, session=session)
    stock = yf.Ticker(symbol)
    news = stock.news[:6]  # Last 6 news items

    all_news = []
    for item in news:
        root =  item["content"]
        all_news += [{
            'title' : root["title"],
            'date' : root["displayTime"],
            'summary' : root["summary"]
        }]
    
    if DEBUG:
        print("NEWSDATA:", all_news, '\n-------')
    else:
        print ("\n-News topics ingested: ")
        for item in all_news:
            print ("-- ", item['title'])
    print ("")

    prompt = PromptTemplate.from_template(
        """You are a proficient trader, needing to analyze the following recent news items for stock symbol {symbol}.
        Your target audience is a group with moderate understanding of stocks and business processes.
        {news}

        Provide:
        1. Overall sentiment
        2. Key developments
        3. Potential impact
        4. Risk factors
        """
    )

    input_data = {
        'symbol': symbol, 
        'news': json.dumps(all_news, indent=2)
    }

    chain =  prompt | LLM_instance
    analysis = chain.invoke(input=input_data)

    if remove_think_tags:
        analysis = re.sub(r"<think>.*?</think>", "", analysis, flags = re.DOTALL)

    answer = {
        "results": {
            "news": {
                "data": all_news,
                "analysis": analysis
            }
        }
    }
    print ("---News analysis: DONE")
    if DEBUG: 
        print ("---End of NEWS: \n", answer, "\n\n")
    return answer

# Final Recommendation Node
# def generate_recommendation(state: InputState) -> Dict:
def generate_recommendation(state: OutputState) -> Dict:
    """Node for final recommendation"""
    symbol = state["symbol"]
    results = state["results"]

    prompt = PromptTemplate.from_template(
        """You are an economic specialist with amazing of knowledge on stock exchange processes. 
        Based on the following analyses for the stock symbol {symbol}, please provide a final recommendation aimed to a group of stock specialists.
        Make sure you use the values as provided in the below analysis, especially regarding the current stock price as mentioned in the technical analysis.

        Technical Analysis:
        {technical}

        Market Analysis:
        {market}

        News Analysis:
        {news}

        Please provide insights regarding at least the following topics:
        1. Final recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        2. Confidence score (1-10)
        3. Key reasons
        4. Risk factors
        5. News summary (concise)
        6. Target price range with clear reasoning and in relation to current market price

        Lastly, please add a Family Guy quote related to this at the end as a closing statement.
        """
    )

    input_data = {
        'symbol': symbol,
        'technical': results["technical"]["analysis"],
        'market': results["market"]["analysis"],
        'news': results["news"]["analysis"]
    }

    chain = prompt | LLM_instance
    final_recommendation = chain.invoke(input=input_data)

    if remove_think_tags:
        final_recommendation = re.sub(r"<think>.*?</think>", "", final_recommendation, flags = re.DOTALL)

    answer = {
        "results": {
            "recommendation": final_recommendation
        }
    }
    if DEBUG: 
        print ("---End of RECOMMENDATION: \n", answer, "\n\n")
    return answer

def create_analysis_graph() -> Graph:
    """Create the analysis workflow graph"""
    # Create workflow graph
    workflow = StateGraph(OverallState, input=InputState, output=OutputState)

    # Add nodes
    workflow.add_node("technical", technical_analysis)
    workflow.add_node("market", market_analysis)
    workflow.add_node("news", news_analysis)
    workflow.add_node("recommendation", generate_recommendation)

    # Define edges
    workflow.add_edge(START, "technical")
    workflow.add_edge(START, "market")
    workflow.add_edge(START, "news")
    workflow.add_edge(["technical", "market", "news"], "recommendation")

    # Set end node
    workflow.add_edge("recommendation", END)

    return workflow.compile()

# AGENT IDEA: Could fetch DJI news for market info? New agent needed for this feeding in to final analysis.
# results = run_analysis("^DJI")

# display(Image(create_analysis_graph().get_graph().draw_mermaid_png()))

# Make into Function: Check on LLM used, if DeepSeek: disable think tags. For now global
remove_think_tags = True
DEBUG = False

symb = "PANW"
#symb = "RBRK"

LLM_instance = setup_llm()
graph = create_analysis_graph()
print ('//--Initiating analysis for stock ticker: ', symb)
result = graph.invoke({"symbol":symb})
print ("\nRESULTS: ", result["results"]["recommendation"])
