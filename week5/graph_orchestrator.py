import os
import sys

# Ensure the 'src' directory is in the system path to reuse Week 3 tools
current_dir = os.path.dirname(os.path.abspath(__file__))
src_parent_dir = os.path.dirname(current_dir)
if src_parent_dir not in sys.path:
    sys.path.append(src_parent_dir)

# Import our REAL tools from Week 3!
try:
    from week3_core.tools import filter_by_source_document, fetch_document_chunk
except ImportError:
    # Fallback mock definitions if run outside the main project workspace
    filter_by_source_document = lambda target_pdf, keyword: "[chunk_03]: Approved remote employees get up to $500 hardware reimbursement."
    fetch_document_chunk = lambda chunk_id: "Specialized Core AI Engineers are eligible for 100% remote status."


class GraphState:
    """The central 'Shared Memory' blackboard that agents read from and write to."""
    def __init__(self, user_query: str):
        self.user_query = user_query
        self.next_step = "ROUTER"
        
        # State variables to accumulate facts
        self.hr_resolved = False
        self.finance_resolved = False
        
        # Extracted data blocks
        self.hr_findings = ""
        self.finance_findings = ""
        self.final_response = ""


# =====================================================================
# THE 3 SPECIALIZED GRAPH AGENTS (NODES)
# =====================================================================

class RouterAgent:
    """Node 1: The Traffic Controller. Decides who handles the task next."""
    def process(self, state: GraphState):
        print("\n🔀 [NODE: Router Agent]")
        print(f"🧠 Inspecting Graph State... HR Resolved: {state.hr_resolved} | Finance Resolved: {state.finance_resolved}")
        
        if not state.hr_resolved:
            print("➡️ Routing Decision: Need to check remote eligibility first. Sending to HR Agent.")
            state.next_step = "HR_AGENT"
        elif state.hr_resolved and not state.finance_resolved:
            print("➡️ Routing Decision: Remote eligibility confirmed! Now sending to Finance Agent for allowance limits.")
            state.next_step = "FINANCE_AGENT"
        else:
            print("➡️ Routing Decision: All specialized data gathered. Routing to Compiler.")
            state.next_step = "COMPILER"


class HRAgent:
    """Node 2: The HR Policy Specialist. Only handles employment status tasks."""
    def process(self, state: GraphState):
        print("\n👔 [NODE: HR Agent]")
        print("🧠 Thought: I will fetch the remote work policy addendum chunk.")
        
        # Executing the specialized HR tool
        observation = fetch_document_chunk(chunk_id="chunk_02")
        print(f"🛠️ [Executing fetch_document_chunk] -> Result found!")
        
        # Update the shared graph state
        state.hr_findings = observation
        state.hr_resolved = True
        state.next_step = "ROUTER"  # Pass control back to the router


class FinanceAgent:
    """Node 3: The Finance Specialist. Only handles budget and reimbursement queries."""
    def process(self, state: GraphState):
        print("\n💰 [NODE: Finance Agent]")
        print("🧠 Thought: I need to query the finance policy specifically for 'reimbursement'.")
        
        # Executing the specialized Finance tool
        observation = filter_by_source_document(target_pdf="Finance_Benefits_2026.pdf", keyword="reimbursement")
        print(f"🛠️ [Executing filter_by_source_document] -> Result found!")
        
        # Update the shared graph state
        state.finance_findings = observation
        state.finance_resolved = True
        state.next_step = "ROUTER"  # Pass control back to the router


class ResponseCompiler:
    """The Final Node: Merges all extracted data into a unified answer."""
    def process(self, state: GraphState):
        print("\n✍️ [NODE: Response Compiler]")
        print("🧠 Thought: Merging HR facts and Finance facts into a clean final response...")
        
        # Synthesize collected state facts
        state.final_response = (
            f"Under the company policy rules:\n"
            f"1. Remote Status: {state.hr_findings}\n"
            f"2. Equipment Allowance: {state.finance_findings}\n"
            f"Result: You qualify for remote status and can safely claim the $500 reimbursement."
        )
        state.next_step = "END"


# =====================================================================
# THE GRAPH RUNTIME ENGINE (ORCHESTRATOR)
# =====================================================================

class MultiAgentGraphEngine:
    """Orchestrates the execution of nodes according to the state transitions."""
    def __init__(self):
        # Registering our nodes
        self.nodes = {
            "ROUTER": RouterAgent(),
            "HR_AGENT": HRAgent(),
            "FINANCE_AGENT": FinanceAgent(),
            "COMPILER": ResponseCompiler()
        }

    def run(self, initial_query: str):
        print(f"🚀 [Starting Multi-Agent Graph Engine]")
        print(f"📥 Input Query: '{initial_query}'")
        
        # Initialize Shared State
        state = GraphState(initial_query)
        
        # The main Graph Execution Loop (The Tick Engine)
        steps = 1
        while state.next_step != "END" and steps <= 10:
            print(f"\n--- Graph State Loop Step {steps} ---")
            current_node_name = state.next_step
            node_instance = self.nodes[current_node_name]
            
            # Execute the designated agent node passing the shared memory state
            node_instance.process(state)
            steps += 1
            
        print("\n🏁 [Graph Execution Completed Successfully]")
        print("=================================================================")
        print(f"✨ FINAL COMPILED RESPONSE:\n{state.final_response}")
        print("=================================================================")


# --- Live Simulation Pipeline ---
if __name__ == "__main__":
    # Same query from Week 3 & 4
    complex_user_query = "As an AI Engineer, am I eligible for remote status and what allowance can I claim?"
    
    graph_engine = MultiAgentGraphEngine()
    graph_engine.run(complex_user_query)