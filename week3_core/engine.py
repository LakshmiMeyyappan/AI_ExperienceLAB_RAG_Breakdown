from src.utils.parsers import parse_thought_and_action
from src.week3_core.tools import TOOL_REGISTRY

class IntegratedAgentEngine:
    def __init__(self, registry: dict, max_loops: int = 5):
        self.registry = registry
        self.max_loops = max_loops
        self.context_history = []

    def _mock_llm_token_stream(self, current_turn: int) -> str:
        """Simulates how a model handles a cross-document, multi-step dependency prompt."""
        if current_turn == 1:
            return (
                "Thought: The user is an AI Engineer asking about remote work options and if the company will cover their home office expenses. This requires data from both workplace policies and financial allowances. I will start by searching the general vector index.\n"
                "Action: query_vector_store({'semantic_query': 'remote AI engineer equipment allowance'})"
            )
        elif current_turn == 2:
            return (
                "Thought: The index search pointed me to three documents: HR_Policy_2026.pdf, AI_Dept_Addendum.pdf, and Finance_Benefits_2026.pdf. I need to explicitly check the financial reimbursements inside Finance_Benefits_2026.pdf.\n"
                "Action: filter_by_source_document({'target_pdf': 'Finance_Benefits_2026.pdf', 'keyword': 'reimbursement'})"
            )
        elif current_turn == 3:
            return (
                "Thought: I see that approved remote workers get a $500 home-office hardware reimbursement. I also know from the previous index step that chunk_02 in AI_Dept_Addendum.pdf determines remote eligibility for AI Engineers. Let me fetch the text of chunk_02 to fully tie the eligibility logic to the financial benefit.\n"
                "Action: fetch_document_chunk({'chunk_id': 'chunk_02'})"
            )
        elif current_turn == 4:
            return (
                "Thought: I have verified both conditions: 1) AI Engineers qualify for 100% remote status via the AI Addendum, and 2) Approved remote status grants a $500 equipment allowance via the Finance file. I am ready to compile the complete answer.\n"
                "Final Answer: Under the 2026 guidelines, AI Engineers are explicitly eligible for 100% fully remote status (subject to quarterly director reviews). Because you qualify for approved remote status, you are fully eligible to claim the one-time home-office equipment reimbursement of up to $500 by submitting itemized receipts through the finance portal."
            )
        return "Final Answer: Evaluation completed."

    def run_research(self, query: str):
        print(f"🚀 Launching Core Agentic RAG Engine")
        print("=" * 70 + "\n")
        print(f" Prompt: '{query}'")
        self.context_history.append({"role": "user", "content": query})
        
        for loop_count in range(1, self.max_loops + 1):
            print(f"\n🔄 [LOOP STEP {loop_count} OF {self.max_loops}]")
            llm_response = self._mock_llm_token_stream(loop_count)
            self.context_history.append({"role": "assistant", "content": llm_response})
            
            if "Final Answer:" in llm_response:
                print(f"\n✅ [CROSS-DOCUMENT SYNTHESIS COMPLETE]:\n{llm_response.split('Final Answer:')[1].strip()}\n")
                break
            
            thought, tool_name, tool_args = parse_thought_and_action(llm_response)
            if thought: print(f"🧠 [Thought Layer]: {thought}")
            
            if tool_name and tool_args is not None:
                print(f"🛠️  [Executing Tool Routing]: {tool_name}(**{tool_args})")
                if tool_name in self.registry:
                    observation = self.registry[tool_name](**tool_args)
                else:
                    observation = f"Error: Tool '{tool_name}' missing from structural registry."
                print(f"👁️  [Observation State]: {observation}")
                self.context_history.append({"role": "user", "content": f"Observation: {observation}"})

if __name__ == "__main__":
    agent = IntegratedAgentEngine(registry=TOOL_REGISTRY)
    agent.run_research("I am an AI Engineer. Can I work fully remote, and will the company pay for my home office setup expenses?")