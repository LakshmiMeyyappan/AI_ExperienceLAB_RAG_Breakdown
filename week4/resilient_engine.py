import os
import sys

# Crucial: Ensure the 'src' directory is in the system path so Python can find 'week3_core'
current_dir = os.path.dirname(os.path.abspath(__file__))
src_parent_dir = os.path.dirname(current_dir)
if src_parent_dir not in sys.path:
    sys.path.append(src_parent_dir)

# Now we import your REAL, actual production tools from Week 3!
try:
    from week3_core.tools import query_vector_store, filter_by_source_document, fetch_document_chunk
except ImportError:
    # Fallback definitions matching your Week 3 screenshots if the file isn't fully populated yet
    print("⚠️ [System Notice]: Direct import from 'week3_core.tools' not found. Using current Week 3 trace specifications.")
    query_vector_store = lambda semantic_query: '[{"id": "chunk_01", "metadata": {"source": "HR_Policy_2026.pdf"}}, {"id": "chunk_02", "metadata": {"source": "AI_Dept_Addendum.pdf"}}, {"id": "chunk_03", "metadata": {"source": "Finance_Benefits_2026.pdf"}}]'
    filter_by_source_document = lambda target_pdf, keyword: '[{"id": "chunk_03", "text": "Approved remote employees are eligible for a one-time home-office equipment reimbursement up to $500."}]'
    fetch_document_chunk = lambda chunk_id: "Specialized Core AI and Machine Learning Engineering teams are eligible for 100% fully remote / Work-From-Home status subject to quarterly director performance reviews."


class MemoryGuardrail:
    """Safeguard 1: Prevents conversation memory from causing token window overflows."""
    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns

    def enforce(self, memory_history: list) -> list:
        if len(memory_history) > self.max_turns:
            print("\n⚠️ [Context Guardrail Triggered]: Pruning old intermediate turns to protect token budget!")
            # Always hold index 0 to preserve the immutable system rules
            system_prompt = memory_history[0]
            pruned_history = [system_prompt] + memory_history[-(self.max_turns - 1):]
            return pruned_history
        return memory_history


class ResilientAgentEngine:
    """Safeguard 2: Dynamic execution engine with built-in Auto-Healing loops."""
    def __init__(self, tool_registry: dict, guardrail: MemoryGuardrail):
        self.tool_registry = tool_registry
        self.guardrail = guardrail
        self.context_history = [
            {"role": "system", "content": "You are a secure AI Agent. Use only registered tools."}
        ]

    def _execute_tool_safely(self, tool_name: str, arguments: dict) -> str:
        """Executes actual imported Week 3 functions inside a protective try-except wall."""
        try:
            # Catching hallucinated tools before execution
            if tool_name not in self.tool_registry:
                raise KeyError(f"The tool '{tool_name}' does not exist in the secure registry.")
            
            # RUNNING THE REAL WEEK 3 CODE DYNAMICALLY!
            return self.tool_registry[tool_name](**arguments)
            
        except Exception as error:
            print(f"❌ [Execution Error Intercepted]: {type(error).__name__} - {str(error)}")
            
            # Constructing the self-healing feedback string
            allowed_tools = list(self.tool_registry.keys())
            auto_healing_feedback = (
                f"Auto-Healing Feedback: Action failed. The tool '{tool_name}' is invalid. "
                f"Your allowed secure action space only consists of: {allowed_tools}. "
                f"Please correct your parameter routing and retry."
            )
            return auto_healing_feedback

    def simulate_agent_turn(self, loop_number: int, thought: str, tool_name: str, tool_args: dict):
        """Executes a full step of the ReAct cognitive loop."""
        print(f"\n🔄 [LOOP STEP {loop_number} OF 5]")
        print(f"🧠 [Thought Layer]: {thought}")
        print(f"🛠️ [Executing Tool Routing]: {tool_name}(**{tool_args})")
        
        # 1. Safely evaluate and run the tool
        observation = self._execute_tool_safely(tool_name, tool_args)
        
        if not observation.startswith("Auto-Healing Feedback"):
            print(f"👁️ [Observation State]: {observation}")
        else:
            print(f"🩹 [Auto-Healing Injected into Memory]")
            
        # 2. Update short-term memory state
        self.context_history.append({"role": "user", "content": f"Observation: {observation}"})
        
        # 3. Trim memory if it exceeds limits
        self.context_history = self.guardrail.enforce(self.context_history)


# --- Execution Execution Pipeline ---
if __name__ == "__main__":
    
    # Map the real imported functions straight from your week3 codebase!
    real_week3_registry = {
        "query_vector_store": query_vector_store,
        "filter_by_source_document": filter_by_source_document,
        "fetch_document_chunk": fetch_document_chunk
    }

    # Initialize the engine
    memory_manager = MemoryGuardrail(max_turns=4)
    engine = ResilientAgentEngine(tool_registry=real_week3_registry, guardrail=memory_manager)

    # ------------------------------------------------------------------------
    # STEP 1: Calling the real 'query_vector_store' from Week 3
    # ------------------------------------------------------------------------
    engine.simulate_agent_turn(
        loop_number=1,
        thought="The user wants to look up remote engineering policies. Running general index scan.",
        tool_name="query_vector_store",
        tool_args={"semantic_query": "remote AI engineer equipment allowance"}
    )

    # ------------------------------------------------------------------------
    # STEP 2: Simulating a mistake to force the Auto-Healing Loop to activate
    # ------------------------------------------------------------------------
    engine.simulate_agent_turn(
        loop_number=2,
        thought="I need to pull the financial boundaries. Let's call the allowance check tool.",
        tool_name="check_allowance_amount",  # Hallucinated tool name!
        tool_args={"policy": "finance"}
    )

    # ------------------------------------------------------------------------
    # STEP 3: Self-correction loop executing a real Week 3 filtering tool
    # ------------------------------------------------------------------------
    engine.simulate_agent_turn(
        loop_number=3,
        thought="My previous tool call failed. Correcting course by using the valid 'filter_by_source_document' tool.",
        tool_name="filter_by_source_document",
        tool_args={"target_pdf": "Finance_Benefits_2026.pdf", "keyword": "reimbursement"}
    )

    # ------------------------------------------------------------------------
    # 🔬 PRINT VERIFICATION PROOF
    # ------------------------------------------------------------------------
    print("\n🔬 --- RAW CONVERSATION MEMORY PROOF ---")
    for index, message in enumerate(engine.context_history):
        print(f"[{index}] Role: {message['role']} | Snippet: {message['content'][:95]}...")