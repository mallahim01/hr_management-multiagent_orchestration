"""agents package"""
from agents.leave_request_agent     import LeaveRequestAgent
from agents.leave_balance_agent     import LeaveBalanceAgent
from agents.company_knowledge_agent import CompanyKnowledgeAgent
from agents.hr_request_agent        import HRRequestAgent
from agents.general_assistant_agent import GeneralAssistantAgent

AGENT_REGISTRY = {
    "LeaveRequestAgent":     LeaveRequestAgent,
    "LeaveBalanceAgent":     LeaveBalanceAgent,
    "CompanyKnowledgeAgent": CompanyKnowledgeAgent,
    "HRRequestAgent":        HRRequestAgent,
    "GeneralAssistantAgent": GeneralAssistantAgent,
}
