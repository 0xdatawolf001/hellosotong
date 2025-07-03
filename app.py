import streamlit as st
import pandas as pd
import requests
import json
import re
from datetime import datetime

# --- App Configuration ---
st.set_page_config(
    page_title="SQD Event Explorer",
    page_icon="🦑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & State ---
CHAINS = {
    "Ethereum": "ethereum-mainnet",
    "Polygon": "polygon-mainnet",
    "BNB Chain": "binance-mainnet",
    "Arbitrum": "arbitrum-one",
    "Optimism": "optimism-mainnet",
    "Avalanche": "avalanche-mainnet",
    "Base": "base-mainnet"
}
BASE_GATEWAY_URL = "https://v2.archive.subsquid.io/network/"

# Initialize session state for the dataframe and dynamic topic inputs
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'topics' not in st.session_state:
    st.session_state.topics = ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]

# --- Helper Functions (Library-Free) ---

@st.cache_data(ttl=300)
def get_gateway_status(gateway_url):
    """Checks the status and height of the selected SQD Network gateway."""
    try:
        response = requests.get(f"{gateway_url}/height")
        response.raise_for_status()
        return {"status": "online", "height": int(response.text)}
    except (requests.exceptions.RequestException, ValueError) as e:
        return {"status": "offline", "error": str(e)}

@st.cache_data
def get_worker_url(gateway_url, block):
    """Fetches the specific worker URL for a given block from the main gateway."""
    try:
        response = requests.get(f"{gateway_url}/{block}/worker")
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        return None

def simplified_decode_log(log, abi=None):
    """Decodes log data using a simplified, library-free approach."""
    decoded = {}
    
    if not abi:
        for i, topic in enumerate(log['topics']):
            decoded[f'topic{i}'] = topic
        decoded['data'] = log['data']
        return decoded

    event_abi = abi['inputs']
    indexed_inputs = [i for i in event_abi if i['indexed']]
    non_indexed_inputs = [i for i in event_abi if not i['indexed']]
    
    for i, item in enumerate(indexed_inputs):
        raw_topic = log['topics'][i + 1]
        if item['type'] == 'address':
            decoded[item['name']] = f"0x{raw_topic[-40:]}"
        elif item['type'].startswith(('uint', 'int')):
            decoded[item['name']] = int(raw_topic, 16)
        else:
            decoded[item['name']] = raw_topic

    if non_indexed_inputs:
        if len(non_indexed_inputs) == 1:
             item = non_indexed_inputs[0]
             if item['type'].startswith(('uint', 'int')):
                 decoded[item['name']] = int(log['data'], 16)
             else:
                 decoded[item['name']] = log['data']
        else:
            decoded['non_indexed_data'] = log['data']
            
    return decoded

def fetch_event_data(gateway_url, contract_address, topic_hashes, abi, start_block, end_block, row_limit):
    """A generator function to fetch event data in chunks."""
    current_block = start_block
    fetched_rows = 0

    while current_block <= end_block:
        worker_url = get_worker_url(gateway_url, current_block)
        if not worker_url:
            st.error(f"Failed to get a worker for block {current_block}. Halting.")
            break
        
        query_payload = {
            "fromBlock": current_block,
            "toBlock": end_block,
            "logs": [{"address": [contract_address.lower()], "topic0": [t.lower() for t in topic_hashes]}],
            "fields": {
                "log": {"topics": True, "data": True, "transactionHash": True, "logIndex": True},
                "block": {"timestamp": True}
            }
        }
        
        try:
            response = requests.post(worker_url, json=query_payload, timeout=60)
            response.raise_for_status()
            batch_data = response.json()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            st.warning(f"Worker query failed: {e}. Trying to continue...")
            break
            
        if not batch_data:
            break

        for block_data in batch_data:
            block_header = block_data.get("header", {})
            block_number = block_header.get("number")
            block_timestamp = block_header.get("timestamp")

            for log in block_data.get("logs", []):
                if fetched_rows >= row_limit: return

                decoded_event = simplified_decode_log(log, abi)
                decoded_event['block_number'] = block_number
                decoded_event['timestamp'] = block_timestamp
                decoded_event['log_index'] = log['logIndex']
                decoded_event['transaction_hash'] = log['transactionHash']
                yield decoded_event
                fetched_rows += 1
        
        last_processed_block = batch_data[-1].get("header", {}).get("number")
        if last_processed_block is None: break
        current_block = last_processed_block + 1
        progress = min(1.0, (current_block - start_block) / (end_block - start_block + 1))
        progress_bar.progress(progress, text=f"Scanned up to block {f'{current_block:,}'}")

# --- Streamlit UI ---

st.title("🦑 SQD Network Event Explorer")
st.caption("Fetch on-chain event data directly from decentralized archival storage.")

# --- Sidebar Configuration ---
st.sidebar.header("1. Select Chain")
selected_chain_name = st.sidebar.selectbox("EVM Chain", options=list(CHAINS.keys()))
gateway_url = BASE_GATEWAY_URL + CHAINS[selected_chain_name]

status_placeholder = st.sidebar.empty()
status_info = get_gateway_status(gateway_url)
latest_block = 0
if status_info["status"] == "online":
    latest_block = status_info['height']
    status_placeholder.success(f"Gateway Online. Height: {latest_block:,}")
else:
    status_placeholder.error(f"Gateway Offline: {status_info.get('error')}")

st.sidebar.header("2. Configure Query")
contract_address = st.sidebar.text_input(
    "Contract Address", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    help="The smart contract address."
)

st.sidebar.subheader("Event Topic0 Hashes")
for i, topic in enumerate(st.session_state.topics):
    st.session_state.topics[i] = st.sidebar.text_input(
        f"Topic {i+1}", value=topic, key=f"topic_{i}",
        help="The unique signature hash of the event."
    )

c1, c2 = st.sidebar.columns(2)
if c1.button("Add Topic", use_container_width=True):
    st.session_state.topics.append("")
    st.rerun()
if c2.button("Remove Last", use_container_width=True):
    if len(st.session_state.topics) > 1:
        st.session_state.topics.pop()
        st.rerun()

st.sidebar.header("3. Decode Results (Optional)")
abi_string = st.sidebar.text_area(
    "Paste Event ABI (for decoding)", height=150, key="abi_input",
    placeholder="Optional: Paste single event ABI...",
    help="Provide the ABI for the specific event to decode results."
)

st.sidebar.header("4. Set Query Range & Limits")
# --- Dynamic Block Range Slider ---
if latest_block > 0:
    default_start = int(latest_block * 0.75)
    selected_range = st.sidebar.slider(
        "Quick Select Block Range",
        min_value=0,
        max_value=latest_block,
        value=(default_start, latest_block),
        step=1000
    )
else: # Fallback if gateway is offline
    selected_range = (18_000_000, 18_010_000)

# --- Number inputs controlled by the slider ---
c1, c2 = st.sidebar.columns(2)
start_block = c1.number_input("Start Block", 0, None, selected_range[0], format="%d")
end_block = c2.number_input("End Block", 0, None, selected_range[1], format="%d")

row_limit = st.sidebar.number_input("Max Rows to Fetch", min_value=1, value=10000, help="The query will stop after fetching this many rows.")

# --- Action Button & Logic ---
if st.sidebar.button("Fetch Event Data", type="primary", use_container_width=True):
    topic_hashes = [t.strip() for t in st.session_state.topics if t.strip()]

    if not (contract_address and topic_hashes and str(start_block) and str(end_block)):
        st.error("Please fill in all required fields: Contract Address, at least one Topic0, Start Block, and End Block.")
    elif any(not re.match(r'^0x[0-9a-fA-F]{64}$', t) for t in topic_hashes):
        st.error("Invalid Topic0 Hash format. Each must be a 66-character hex string (including 0x).")
    elif end_block < start_block:
        st.error("End Block cannot be before Start Block.")
    else:
        event_abi_json = None
        if abi_string:
            try:
                parsed_json = json.loads(abi_string)
                event_abi_json = parsed_json[0] if isinstance(parsed_json, list) else parsed_json
            except json.JSONDecodeError:
                st.error("Invalid ABI JSON provided for decoding.")
                st.stop()

        st.session_state.df = pd.DataFrame()
        with st.container(border=True):
            st.write(f"**Contract:** `{contract_address}`")
            st.write(f"**Topic0 Hashes:**")
            st.code('\n'.join(topic_hashes))

        progress_bar = st.progress(0, text="Starting scan...")
        
        try:
            with st.spinner("Querying the SQD data lake... This may take a moment."):
                event_generator = fetch_event_data(
                    gateway_url, contract_address, topic_hashes, event_abi_json, start_block, end_block, row_limit
                )
                results = list(event_generator)

            if results:
                st.session_state.df = pd.DataFrame(results).fillna("")
                if 'timestamp' in st.session_state.df.columns:
                    st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['timestamp'], unit='s', errors='coerce')
                
                cols_order = ['block_number', 'timestamp', 'log_index', 'transaction_hash']
                existing_cols = [col for col in cols_order if col in st.session_state.df.columns]
                other_cols = [col for col in st.session_state.df.columns if col not in existing_cols]
                st.session_state.df = st.session_state.df[existing_cols + other_cols]

                progress_bar.progress(1.0, text="Scan complete!")
            else:
                progress_bar.progress(1.0, text="Scan complete!")
                st.warning("No events found for the specified criteria in the given block range.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            progress_bar.empty()

# --- Display Results ---
if not st.session_state.df.empty:
    st.subheader("Query Results")
    st.write(f"Displaying **{len(st.session_state.df)}** of {row_limit} max rows.")
    st.dataframe(st.session_state.df)

    csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name=f"{selected_chain_name}_{contract_address[:8]}_events.csv",
        mime='text/csv',
    )
else:
    st.info("Configure your query in the sidebar and click 'Fetch Event Data' to begin.")