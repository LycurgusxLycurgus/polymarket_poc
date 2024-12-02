import os
import json
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from openai import OpenAI
import time
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Initialize OpenAI client as shown in documentation
api_key = os.getenv("GLHF_API_KEY")
if not api_key:
    raise ValueError("GLHF_API_KEY environment variable not set.")

client = OpenAI(
    api_key=api_key,
    base_url="https://glhf.chat/api/openai/v1"
)

def transform_query_to_keywords(query):
    """Transform user query into search keywords using Gemma LLM."""
    system_prompt = """You are a search keyword extractor. Your task is to transform a user's query into a single relevant keyword 
    that can be used to search on Polymarket. You must respond in the following JSON format:
    {
        "keyword": "your extracted keyword here" (ONLY ONE WORD)
    }
    Be concise and focus on the most important concept. Return ONLY the JSON, nothing else."""
    
    user_prompt = f""" important: Extract ONLY ONE keyword that is most relevant for searching. Just return the main concept as the keyword.
      Query: {query}

Note 1: You must respond in the following JSON format:
{{
    "keyword": "your extracted keyword here" (ONLY ONE WORD)
}}

Note 2: Extract ONLY ONE keyword that is most relevant for searching. Just return the main concept as the keyword.

Note 3: Return ONLY the JSON, no other text, no markdown formatting."""
    
    try:
        print("\nDEBUG - Sending request to Gemma API:")
        completion = client.chat.completions.create(
            model="hf:google/gemma-2-9b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        print("\nDEBUG - Raw API Response:")
        print(completion.choices[0].message)
        
        content = completion.choices[0].message.content
        print("\nDEBUG - Content:", content)
        
        try:
            result = json.loads(content)
            keyword = result.get("keyword", "").strip()
            # Ensure we only take the first word if multiple are returned
            keyword = keyword.split()[0] if keyword else None
            return keyword
        except json.JSONDecodeError as e:
            print(f"DEBUG - JSON parsing error: {e}")
            return None
            
    except Exception as e:
        print(f"Error transforming query: {e}")
        return None

def analyze_market_relevance(query, markets):
    """Analyze market relevance using Gemma LLM."""
    system_prompt = """You are a market relevance analyzer. Given a user's original query and a list of market data, 
    determine which markets are relevant to the query. You must respond in the following JSON format:
    {
        "relevant_market": [
            "market title 1"
        ]
    }
    Return ONLY the JSON with the list of relevant market titles, nothing else. Only return one market title, the most relevant one to the query."""
    
    try:
        print("\nDEBUG - Sending request to Gemma API:")
        completion = client.chat.completions.create(
            model="hf:google/gemma-2-9b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\nMarkets: {json.dumps(markets, indent=2)}"}
            ]
        )
        
        print("\nDEBUG - Raw API Response:")
        print(completion.choices[0].message)
        
        content = completion.choices[0].message.content
        print("\nDEBUG - Content:", content)
        
        try:
            result = json.loads(content)
            return result.get("relevant_market", [])
        except json.JSONDecodeError as e:
            print(f"DEBUG - JSON parsing error: {e}")
            return []
            
    except Exception as e:
        print(f"Error analyzing market relevance: {e}")
        return []

def analyze_final(query, market_details):
    """Final analysis using Nemotron LLM."""
    system_prompt = """ IMPORTANT: Return ONLY the JSON, nothing else, no markdown formatting nor backticks, nor bold asterisks or other formatting.
    You are a financial market analyst specializing in prediction markets. Given a user's query and detailed market data, 
    provide a thorough analysis in formal but simple language. You must respond in the following JSON format:
    {
        "analysis": {
            "bet_description": "What the bet is about and what outcomes are possible",
            "probabilities": "Current probabilities and what they mean",
            "volume_and_liquidity": "Trading volume and market liquidity analysis",
            "opportunities": "Most profitable opportunities if any",
            "risks": "Risks and uncertainties",
            "additional_info": "Any other relevant information",
            "summary": "A concise summary of the analysis"
        }
    }
    Make your explanation detailed but easy to understand for someone who can only see your description.
    Return ONLY the JSON, nothing else, no markdown formatting nor backticks, nor bold asterisks or other formatting. """
    
    try:
        print("\nDEBUG - Sending request to Nemotron API:")
        completion = client.chat.completions.create(
            model="hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\nMarket Details: {json.dumps(market_details, indent=2)} ; Return ONLY the JSON, nothing else, no markdown formatting nor backticks, nor bold asterisks or other formatting."}
            ]
        )
        
        print("\nDEBUG - Raw API Response:")
        print(completion.choices[0].message)
        
        content = completion.choices[0].message.content
        print("\nDEBUG - Content:", content)
        
        try:
            result = json.loads(content)
            return result.get("analysis")
        except json.JSONDecodeError as e:
            print(f"DEBUG - JSON parsing error: {e}")
            return None
            
    except Exception as e:
        print(f"Error in final analysis: {e}")
        return None

@dataclass
class Outcome:
    title: str
    volume: str
    percentage: str
    buy_yes_price: str
    buy_no_price: str

def get_element_text(driver, selector, timeout=15):
    """Helper function to get element text safely"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return element.text.strip()
    except Exception as e:
        print(f"Error getting text for selector {selector}: {str(e)}")
        return None

def extract_market_details(driver, market_card):
    """Extract title, volume, and end date from market card"""
    try:
        # Use the new selectors for market details
        title = get_element_text(driver, ".c-dqzIym.c-gYmnSl.c-dqzIym-fxyRaa-color-normal.c-dqzIym-cTvRMP-spacing-normal.c-dqzIym-dxJWYY-weight-bold")
        volume = get_element_text(driver, ".c-dqzIym.c-dqzIym-fxyRaa-color-normal.c-dqzIym-cTvRMP-spacing-normal.c-dqzIym-jalaKP-weight-normal.c-dqzIym-hzzdKO-size-md.c-dqzIym-iUwoCw-css")
        end_date = get_element_text(driver, "p.c-dqzIym.c-dqzIym-fxyRaa-color-normal.c-dqzIym-cTvRMP-spacing-normal.c-dqzIym-jalaKP-weight-normal.c-dqzIym-hzzdKO-size-md.c-dqzIym-idjDWCM-css.c-PJLV span")
        
        return {
            "title": title,
            "volume": volume,
            "end_date": end_date
        }
    except Exception as e:
        print(f"Error extracting market details: {str(e)}")
        return None

def extract_outcome_data(driver, market_card):
    """Extract outcome data using the working approach from test_scraper"""
    try:
        outcomes = []
        base_path = "body > div:nth-child(1) > div:nth-child(3) > main:nth-child(3) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2)"
        
        # First try to find multiple outcomes
        try:
            # Original multiple outcomes logic
            for outcome_index in range(3, 5):
                try:
                    outcome_base = f"{base_path} > div:nth-child({outcome_index})"
                    
                    # Construct selectors for each element
                    selectors = {
                        "title": f"{outcome_base} > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > p:nth-child(1)",
                        "volume": f"{outcome_base} > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > p:nth-child(1)",
                        "percentage": f"{outcome_base} > div:nth-child(2) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2) > p:nth-child(1)",
                        "buy_yes": f"{outcome_base} > div:nth-child(3) > div:nth-child(1)",
                        "buy_no": f"{outcome_base} > div:nth-child(3) > div:nth-child(2)"
                    }
                    
                    print(f"\nProcessing Outcome {outcome_index-2}")
                    
                    try:
                        title_element = driver.find_element(By.CSS_SELECTOR, selectors["title"])
                        driver.execute_script("arguments[0].scrollIntoView(true);", title_element)
                        time.sleep(1)
                    except Exception as scroll_error:
                        print(f"Scroll error: {scroll_error}")
                        raise  # Propagate the error to trigger single outcome logic
                    
                    # Extract all data
                    title = get_element_text(driver, selectors["title"])
                    volume = get_element_text(driver, selectors["volume"])
                    percentage = get_element_text(driver, selectors["percentage"])
                    buy_yes = get_element_text(driver, selectors["buy_yes"])
                    buy_no = get_element_text(driver, selectors["buy_no"])
                    
                    if title:
                        outcome = Outcome(
                            title=title,
                            volume=volume,
                            percentage=percentage,
                            buy_yes_price=buy_yes,
                            buy_no_price=buy_no
                        )
                        outcomes.append(outcome)
                        print(f"Successfully extracted outcome {outcome_index-2}")
                    
                except Exception as e:
                    print(f"Error processing outcome {outcome_index-2}: {str(e)}")
                    raise  # Propagate the error to trigger single outcome logic
                    
        except Exception as multi_outcome_error:
            print("Attempting to process single outcome case...")
            # Single outcome selectors
            single_selectors = {
                "percentage": ".c-dqzIym.c-dqzIym-fxyRaa-color-normal.c-dqzIym-cTvRMP-spacing-normal.c-dqzIym-jalaKP-weight-normal.c-dqzIym-idXFcBQ-css",
                "type": ".c-dqzIym.c-dqzIym-fxyRaa-color-normal.c-dqzIym-cTvRMP-spacing-normal.c-dqzIym-jalaKP-weight-normal.c-dqzIym-euRFQq-size-sm.c-dqzIym-igYhVgt-css",
                "buy_yes": ".c-gBrBnR.c-dETnmA.c-gBrBnR-iicKxNF-css",
                "buy_no": ".c-gBrBnR.c-dETnmA.c-gBrBnR-iggWduY-css"
            }
            
            # Get market title (reuse from market details)
            title = get_element_text(driver, ".c-dqzIym.c-gYmnSl.c-dqzIym-fxyRaa-color-normal.c-dqzIym-cTvRMP-spacing-normal.c-dqzIym-dxJWYY-weight-bold")
            percentage = get_element_text(driver, single_selectors["percentage"])
            outcome_type = get_element_text(driver, single_selectors["type"])
            buy_yes = get_element_text(driver, single_selectors["buy_yes"])
            buy_no = get_element_text(driver, single_selectors["buy_no"])
            
            if title and percentage:
                outcome = Outcome(
                    title=title,
                    volume="N/A",  # Volume doesn't apply for single outcome
                    percentage=percentage,
                    buy_yes_price=buy_yes,
                    buy_no_price=buy_no
                )
                outcomes.append(outcome)
                print("Successfully extracted single outcome")
        
        return [vars(outcome) for outcome in outcomes]  # Convert dataclass instances to dicts
        
    except Exception as e:
        print(f"Error in outcome extraction: {str(e)}")
        return []

def setup_driver():
    """Setup Chrome driver with tablet-like dimensions in headless mode"""
    chrome_options = Options()
    
    # Headless mode configuration
    chrome_options.add_argument("--headless=new")  # New headless mode
    chrome_options.add_argument("--window-size=820,1180")  # Tablet-like dimensions
    chrome_options.add_argument("--disable-gpu")  # Required for some systems
    chrome_options.add_argument("--no-sandbox")  # Required for some systems
    chrome_options.add_argument("--disable-dev-shm-usage")  # Memory optimization
    
    # Create and return the driver
    return webdriver.Chrome(options=chrome_options)

def click_market_card(driver, card_index):
    """Click on a specific market card using precise CSS selector"""
    try:
        # Construct the selector for the specific card's title link
        card_selector = f"body > div:nth-child(1) > div:nth-child(3) > main:nth-child(3) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2) > div:nth-child(4) > div:nth-child(1) > div:nth-child({card_index}) > div:nth-child(1) > div:nth-child(2) > a:nth-child(1) > p:nth-child(1)"
        
        # Find and scroll to the element
        card_element = driver.find_element(By.CSS_SELECTOR, card_selector)
        driver.execute_script("arguments[0].scrollIntoView(true);", card_element)
        time.sleep(1)  # Wait for scroll
        
        # Click using JavaScript to avoid any overlay issues
        driver.execute_script("arguments[0].click();", card_element)
        time.sleep(3)  # Wait for page load
        
        return True
    except Exception as e:
        print(f"Error clicking market card {card_index}: {str(e)}")
        return False

def main():
    driver = None
    try:
        # Test query
        user_query = "starship launches in 2024"
        
        print("Initializing WebDriver...")
        driver = setup_driver()
        
        # Step 1: Transform query to keywords using LLM
        print("\nTransforming query to keywords...")
        search_keywords = transform_query_to_keywords(user_query)
        if not search_keywords:
            print("Failed to transform query to keywords")
            return
        print(f"Search keywords: {search_keywords}")
        
        # Navigate to the search URL with transformed keywords
        search_url = f"https://polymarket.com/markets?_q={search_keywords}"
        print(f"\nNavigating to: {search_url}")
        driver.get(search_url)
        time.sleep(5)  # Wait for page load
        
        # Find market cards using the precise selectors
        cards_selector = "div:nth-child(1) > div:nth-child(3) > main:nth-child(3) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div:nth-child(2) > div:nth-child(4) > div:nth-child(1) > div"
        market_cards = driver.find_elements(By.CSS_SELECTOR, cards_selector)
        print(f"\nFound {len(market_cards)} market cards")
        
        # Get titles for relevance analysis
        market_titles = []
        for idx, card in enumerate(market_cards, 1):
            try:
                title_selector = "div:nth-child(1) > div:nth-child(2) > a:nth-child(1) > p:nth-child(1)"
                title = card.find_element(By.CSS_SELECTOR, title_selector).text
                market_titles.append({"title": title, "index": idx})
                print(f"Found market {idx}: {title}")
            except Exception as e:
                print(f"Error extracting card title: {str(e)}")
                continue
        
        # Step 2: Analyze market relevance using LLM
        print("\nAnalyzing market relevance...")
        llm_response = analyze_market_relevance(user_query, market_titles)
        
        # Parse the LLM response to get relevant markets
        relevant_markets = []
        
        # Handle the response as a list
        relevant_titles = llm_response if isinstance(llm_response, list) else []
        
        # Match the returned titles with our market_titles list
        for title in relevant_titles:
            for market in market_titles:
                if title.lower().strip() == market["title"].lower().strip():
                    relevant_markets.append(market)
                    break
        
        if not relevant_markets:
            print("No relevant markets found")
            print("Available market titles were:", [m["title"] for m in market_titles])
            return
            
        print(f"\nFound {len(relevant_markets)} relevant markets:")
        for market in relevant_markets:
            print(f"- {market['title']}")
        
        # Process relevant markets
        results = []
        for market in relevant_markets:
            try:
                # Click on the market card
                if not click_market_card(driver, market['index']):
                    continue
                
                # Extract market details
                details = extract_market_details(driver, driver)
                if not details:
                    continue
                
                # Extract outcomes
                outcomes = extract_outcome_data(driver, driver)
                details['outcomes'] = outcomes
                
                results.append(details)
                print(f"\nProcessed market: {details['title']}")
                
                # Go back to search results
                driver.back()
                time.sleep(3)
                
            except Exception as e:
                print(f"Error processing market card: {str(e)}")
                driver.get(search_url)
                time.sleep(3)
                continue
        
        # Step 3: Final analysis using Nemotron
        print("\nPerforming final analysis...")
        analysis = analyze_final(user_query, results)
        
        if analysis:
            final_results = {
                "query": user_query,
                "search_keywords": search_keywords,
                "market_details": results,
                "analysis": analysis
            }
            
            output_file = 'market_analysis.json'
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=4)
            print(f"\nResults saved to {output_file}")
            
            # Print analysis summary in paragraphs
            print("\nAnalysis Summary:")
            print("\n" + analysis['bet_description'])
            print("\n" + analysis['probabilities'])
            print("\n" + analysis['volume_and_liquidity'])
            print("\n" + analysis['opportunities'])
            print("\n" + analysis['risks'])
            print("\n" + analysis['additional_info'])
            print("\n" + analysis['summary'])
            
            return final_results
        else:
            print("Failed to generate final analysis")
            return None
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    results = main()
    if results:
        print("\nFinal Results:", json.dumps(results, indent=2)) 