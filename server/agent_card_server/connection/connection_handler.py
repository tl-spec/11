from flask_socketio import SocketIO
from agent_card_server.Environment.environment import InformationSeekingEnvironment
# from agent_card_server.Environment.environmentChatFlare import InformationSeekingEnvironment
# from agent_card_server.Agent.IR_agent_new import IRAgent
from agent_card_server.AgentChatFlare.IRAgent import create_ir_agent
from typing import Dict, Any, List
from pydantic import BaseModel
from agent_card_server.AgentChatFlare.IRAgent import IRAgent
from agent_card_server.connection.firebase_handler import create_user_study_section, update_user_study_section

class ClientConnectionHandler: 
    """
    Connection handler for a specific client for agent server.
    TODO: add base class for this
    """
    def __init__(self, socket: SocketIO, sid: str, namespace: str = "/agent_communication"): 
        self.socketApp = socket
        self.sid = sid
        self.namespace = namespace
        self.registedEnvs: Dict[str, InformationSeekingEnvironment] = {}
        self.registedAgents: Dict[str, IRAgent:IRAgent] = {}  
        self.connected = True 
        self.in_user_study = False 
        self.user_study_config = {}
        

    def disconnect(self): 
        self.connected = False
        self.sid = None 
        self.namespace = None 
        for agent in self.registedAgents.values():
            agent.stop_workflow()
            
            
    def start_user_study(self, user_study_id: str, uid: str, projectId: str=None):
        self.in_user_study = True 
        self.user_study_config = {
            "user_study_id": user_study_id,
            "uid": uid,
            "projectId": projectId,
        }
        create_user_study_section(user_study_id, uid, projectId)

    def update_user_study(self, user_study_id: str, uid: str, projectId: str=None):
        if not self.in_user_study:
            self.in_user_study = True
        self.user_study_config = {
            "user_study_id": user_study_id,
            "uid": uid,
            "projectId": projectId
        }
        update_user_study_section(user_study_id, uid, projectId)
        
    
    def end_user_study(self):
        self.in_user_study = False 
        self.user_study_config = {}

    def emit(self, event: str, data: dict):
        """
        emit an event to the specific client
        """
        if self.connected and self.socketApp is not None:
            self.socketApp.emit(event, data, room=self.sid, namespace=self.namespace)

    def initializeEnv(self, envName: str, envId: str, path: str, cardId: str) -> Dict:
        self.registedEnvs[cardId] = InformationSeekingEnvironment()
        print(f"logging: set requestedEnv for {self.sid} with {envName}")
        print(f"start loading env {envName}")
        self.registedEnvs[cardId].registerClientCardId(cardId)
        return self.registedEnvs[cardId].load_existing_env(path, envName, envId)

    def initializeAgent(self, agentId: str, client_handler=None):
        if agentId not in self.registedAgents:
            self.registedAgents[agentId] = create_ir_agent(agentId, self)
        else: 
            print(f"Agent {agentId} already exists")
        if client_handler is not None:
            self.registedAgents[agentId].client_handler = client_handler
        
    def addAgentToEnv(self, agentId: str, cardId: str):
        self.registedEnvs[cardId].registerAgent(self.registedAgents[agentId])
        self.registedAgents[agentId].assign_environment(self.registedEnvs[cardId])

    def initializeSubEnv(self, envName: str, sourceCardId: str, targetCardId: str, documentIds: List[str]):
        self.registedEnvs[targetCardId] = InformationSeekingEnvironment()
        self.registedEnvs[targetCardId].registerClientCardId(targetCardId)
        self.registedEnvs[targetCardId].create_sub_env(self.registedEnvs[sourceCardId], documentIds, envName, targetCardId)
        print(f"***logging: set requestedEnv with {targetCardId}")

    def relayoutEnv(self, cardId: str, task: str):
        return self.registedEnvs[cardId].relayoutProjection(task, None, self, cardId)

    def emitTaskEnvLayoutComputationProgress(self, data: Dict[str, Any]):
        self.emit("task_env_layout_computation_progress", data)

    def emitAgentWorkingProgress(self, data: Dict[str, Any]):
        self.emit("agent_working_progress", data)
        append_to_file("./agent_card_server/logs/agent_working_progress.log", str(data))

    def emitCardComputationMeta(self, data):
        self.emit("card_computation_meta", data)

    def emitOverallSummarizationProgress(self, data: Dict[str, Any]):
        self.emit("overall_summarization_progress", data)
        append_to_file("./agent_card_server/logs/agent_working_progress.log", str(data))
    
    def emitAgentFinalResult(self, data: Dict[str, Any]):
        # self.emit("agent_final_result", data)
        append_to_file("./agent_card_server/logs/agent_working_progress.log", str(data))
    
    @property
    def is_connected(self): 
        return self.connected

def registerClient(sessions: Dict[str, ClientConnectionHandler], sid: str, socket: SocketIO):
    sessions[sid] = ClientConnectionHandler(socket, sid)
    return sessions[sid]

def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:  
            file.write(content + '\n')  
        # print(f"Successfully appended to {file_path}")
    except Exception as e:
        print(f"Failed to append to {file_path}: {e}")