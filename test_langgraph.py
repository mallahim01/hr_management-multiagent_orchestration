"""Quick test: import LangGraphOrchestrator, build the graph, save image."""
import sys, os
sys.path.insert(0, ".")
os.makedirs("data", exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

from core.llm_wrapper import LLMWrapper
from database.db import DatabaseManager
from database.schema import initialize_database

db = DatabaseManager("data/hr_demo.db")
initialize_database(db, 1)

llm = LLMWrapper(model="gpt-4o-mini", max_retries=3, temperature=0.7)

print("Building LangGraph orchestrator...")
from orchestration.langgraph_adapter import LangGraphOrchestrator
orch = LangGraphOrchestrator(llm, db, history_size=3)
print("Graph compiled successfully!")

# Also try saving to a custom path
path = orch.save_graph_image("data/langgraph_flow.png")
print(f"Graph image: {path}")

# Check if file exists and has content
if os.path.isfile("data/langgraph_flow.png"):
    size = os.path.getsize("data/langgraph_flow.png")
    print(f"Image file size: {size} bytes")
    print("PASS")
else:
    print("FAIL: image file not created")
