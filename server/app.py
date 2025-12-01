from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO
from agent_card_server.connection.connection_handler import ClientConnectionHandler, registerClient
from agent_card_server.utils.env_loader import load_environment_variables
from agent_card_server.connection.firebase_handler import list_all_projects, download_projects_from_firebase, download_ris
from agent_card_server.cache_management.manage import CacheManager
from agent_card_server.Parser.risparser import process_ris_pipeline
from typing import Dict
import json
import uuid
import time
import os
from dotenv import load_dotenv

DEBUG = False
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config.from_object(__name__)

session: Dict[str, ClientConnectionHandler] = {}

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})
socketio = SocketIO(app, cors_allowed_origins="*",
                    ping_interval=25, ping_timeout=60)

availableEnvPathMapping = list_all_projects()

load_environment_variables()

# generate a request for hello message 
@app.route('/hello', methods=['GET'])
def hello():
    global availableEnvPathMapping
    availableEnvPathMapping = list_all_projects()
    return jsonify({'message': 'Hello, World!'})

@socketio.on('message', namespace='/agent_communication')
def handle_message(message):
    global availableEnvPathMapping
    print(f"{request.sid} said {message}")
    print("update listing project")
    availableEnvPathMapping = list_all_projects()
    clientSession = registerClient(session, request.sid, socketio)
    clientSession.emit("message", {"data": f"{request.sid} said {message}"})

@socketio.on('disconnect', namespace='/agent_communication')
def handle_disconnect():
    client_sid = request.sid
    print(f"Client {client_sid} disconnected")
    # Clean up session
    if client_sid in session:
        session[client_sid].disconnect()  
        del session[client_sid]

        
@app.route('/createUserStudySection', methods=['POST'])
def createUserStudySection():
    data = request.get_json()
    user_study_id = data.get('user_study_id')
    uid = data.get('uid')
    projectId = data.get('projectId')
    client_sid = data.get('sid')
    if session.get(client_sid) is None:
        return Response(status=404)
    session[client_sid].start_user_study(user_study_id, uid, projectId)
    return Response(status=200)

@app.route('/updateUserStudySection', methods=['POST'])
def updateUserStudySection():
    data = request.get_json()
    user_study_id = data.get('user_study_id')
    uid = data.get('uid')
    projectId = data.get('projectId')
    client_sid = data.get('sid')
    if session.get(client_sid) is None:
        return Response(status=404)
    session[client_sid].update_user_study(user_study_id, uid, projectId)
    return Response(status=200)

@app.route('/endUserStudySection', methods=['POST'])
def endUserStudySection():
    data = request.get_json()
    client_sid = data.get('sid')
    if session.get(client_sid) is None:
        return Response(status=404)
    session[client_sid].end_user_study()
    return Response(status=200)

    
@app.route('/proceeAndRetrieveEnvData', methods=['POST'])
def proceeAndRetrieveEnvData():
    data = request.get_json()
    ris_path = data['ris_path']#获取上传的RIS文件路径
    print(ris_path)
    # 调用firebase_handler中的方法
    local_path = download_ris(ris_path)#从云端下载RIS文件到本地
    projectName = ris_path.split('/')[-1].split('.')[0]#提取项目名（文件名）
    #核心：解析RIS文件并提取文献信息
    projectId = process_ris_pipeline(local_path, projectName)
    #后续：下载项目数据并初始化环境……
    projectPath = download_projects_from_firebase(projectId)
    client_sid = data.get("sid")
    clientCardId = data.get("cardId")
    envData = session[client_sid].initializeEnv(projectName, projectId, path=projectPath, cardId=clientCardId)
    envData['projectId'] = projectId
    envData['projectName'] = projectName
    return Response(json.dumps(envData), status=200, mimetype='application/json')

@app.route('/retrieveEnvData', methods=['POST'])
def retrieveEnvData():
    global availableEnvPathMapping
    data = request.get_json()
    print(data)
    requestedEnvId = data['envId']
    if requestedEnvId not in availableEnvPathMapping:
        availableEnvPathMapping = list_all_projects()
        
    if requestedEnvId not in availableEnvPathMapping:
        print(f"requested env {requestedEnvId} not found")
        return Response(status=404)
    else: 
        print("here") 
        print(availableEnvPathMapping)
        projectPath = download_projects_from_firebase(requestedEnvId)
        print("downloaded")
        print(projectPath)
        requestedEnvName = availableEnvPathMapping[requestedEnvId]["projectName"]
        client_sid = data.get("sid")
        clientCardId = data.get("cardId")
        envData = session[client_sid].initializeEnv(requestedEnvName, requestedEnvId, path=projectPath, cardId=clientCardId)
        return Response(json.dumps(envData), status=200, mimetype='application/json')

@app.route('/retrieveEnvWithTask', methods=['POST'])
def layoutEnvWithTask():
    data = request.get_json()
    client_sid = data.get("sid")
    envName = data.get("envName")
    task = data.get("task")
    cardId = data.get("cardId")
    if session.get(client_sid) is None or (cardId not in session[client_sid].registedEnvs):
        return Response(status=404)
    envDataWithTaskLayout = session[client_sid].registedEnvs[cardId].assign_workspace_with_task(task, None, session[client_sid], cardId)
    return Response(json.dumps(envDataWithTaskLayout), status=200, mimetype='application/json')

@app.route('/registerEnvTask', methods=['POST'])
def registerEnvTask(): 
    data = request.get_json()
    client_sid = data.get("sid") 
    cardId = data.get("cardId") 
    task = data.get("task") 
    envName = data.get("envName")
    sourceCardId = data.get("from_source_cardId")
    documentGroupIds = data.get("document_group_ids")
    if session.get(client_sid) is None:#(cardId not in session[client_sid].registedEnvs):
        return Response(status=404)
    print(f"logging: register env task for card {cardId}")
    session[client_sid].initializeSubEnv(envName, sourceCardId, cardId, documentGroupIds)
    return Response(status=200)


@app.route('/relayoutProjection', methods=['POST'])
def relayoutProjection():
    data = request.get_json()
    client_sid = data.get("sid")
    cardId = data.get("cardId")
    task = data.get("task")
    if session.get(client_sid) is None or (cardId not in session[client_sid].registedEnvs):
        return Response(status=404)
    print(f"logging: relayout projection for card {cardId} with task {task}")
    envDataWithTaskLayout = session[client_sid].relayoutEnv(cardId, task)
    return Response(json.dumps(envDataWithTaskLayout), status=200, mimetype='application/json')

@app.route('/registerAgentInEnvCard', methods=['POST'])
def registerAgentInEnvCard():
    data = request.get_json()
    client_sid = data.get("sid")
    cardId = data.get("cardId")
    agentId = data.get("agentId")
    if session.get(client_sid) is None or (cardId not in session[client_sid].registedEnvs):
        return Response(status=404)
    print(f"logging: register agent {agentId} in env card for card {cardId}")
    session[client_sid].initializeAgent(agentId, client_handler=session[client_sid])
    session[client_sid].addAgentToEnv(agentId, cardId)
    return Response(status=200)

    
@app.route('/registerAgent', methods=['POST'])
def registerAgent():
    data = request.get_json()
    client_sid = data.get("sid")
    agentId = data.get("agentId")
    if session.get(client_sid) is None:
        print(session.get(client_sid) is None)
        print(cardId not in session[client_sid].registedEnvs)
        print(cardId)
        print(session[client_sid].registedEnvs.keys())
        return Response(status=404)
    print(f"logging: register agent {agentId} but not in env card")
    session[client_sid].initializeAgent(agentId, client_handler=session[client_sid])
    # session[client_sid].addAgentToEnv(agentId, cardId)
    return Response(status=200)

@app.route('/agentUpdateMaxIterAllowed', methods=['POST'])    
def agentUpdateMaxIterAllowed():
    data = request.get_json()
    client_sid = data.get("sid")
    agent_id = data.get("agentId")
    maxIterAllowed = data.get("maxIterAllowed")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: agent {agent_id} update research question")
    session[client_sid].registedAgents[agent_id].update_traverse_max_depth(maxIterAllowed, session[client_sid])
    return Response(status=200)


@app.route('/agentUpdateCurrentModel', methods=['POST'])    
def agentUpdateCurrentModel():
    data = request.get_json()
    client_sid = data.get("sid")
    agent_id = data.get("agentId")
    currentModel = data.get("currentModel")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: agent {agent_id} update research question")
    session[client_sid].registedAgents[agent_id].update_workflow_model(currentModel, session[client_sid])
    return Response(status=200)

    

@app.route('/agentUpdateResearchQuestion', methods=['POST'])
def agentUpdateResearchQuestion():
    data = request.get_json()
    client_sid = data.get("sid")
    agent_id = data.get("agentId")
    researchQuestion = data.get("researchQuestion")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: agent {agent_id} update research question")
    session[client_sid].registedAgents[agent_id].update_research_question(researchQuestion, session[client_sid])
    return Response(status=200)

@app.route('/agentUpdateDetailedFocus', methods=['POST'])
def agentUpdateDetailedFocus():
    data = request.get_json()
    client_sid = data.get("sid")
    agent_id = data.get("agentId")
    detailedFocus = data.get("detailedFocus")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: agent {agent_id} update detailed focus")
    session[client_sid].registedAgents[agent_id].update_user_specified_requirement(detailedFocus, session[client_sid])
    return Response(status=200)

@app.route('/agentUpdateInCriteria', methods=['POST'])
def agentUpdateInCriteria():
    data = request.get_json()
    client_sid = data.get("sid")
    agent_id = data.get("agentId")
    inCriteria = data.get("inCriteria")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: agent {agent_id} update in criteria")
    session[client_sid].registedAgents[agent_id].update_inclusion_exclusion_critera(inCriteria, session[client_sid])
    return Response(status=200)

@app.route('/agentUpdateSummarizationRequirement', methods=['POST'])
def agentUpdateSummarizationRequirement():
    data = request.get_json()
    client_sid = data.get("sid")
    agent_id = data.get("agentId")
    summarizationRequirement = data.get("summarizationRequirement")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: agent {agent_id} update summarization requirement")
    session[client_sid].registedAgents[agent_id].update_summarization_requirement(summarizationRequirement, session[client_sid])
    return Response(status=200)


@app.route('/communicateWithAgent', methods=['POST']) 
def communicateWithAgent():
    data = request.get_json()
    client_sid = data.get("sid")
    agentId = data.get("agentId")
    command = data.get("command")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: communicate with agent {agentId}")
    session[client_sid].registedAgents[agentId].communicate_with_agent(command)
    return Response(status=200)

@app.route('/startAgentWorkingOnEnvCard', methods=['POST'])
def startAgentWorkingOnEnvCard():
    data = request.get_json()
    client_sid = data.get("sid")
    cardId = data.get("cardId")
    agentId = data.get("agentId")
    task = data.get("task") 
    if session.get(client_sid) is None or (cardId not in session[client_sid].registedEnvs):
        return Response(status=404)
    print(f"logging: start agent working on env card for card {cardId}")
    session[client_sid].registedAgents[agentId].assign_task(task=task, environment=session[client_sid].registedEnvs[cardId], workspace=None, client_handler=session[client_sid])
    return Response(status=200)

@app.route('/summarizeBetweenAgents', methods=['POST'])
def summarizeBetweenAgents():
    data = request.get_json()
    client_sid = data.get("sid")
    globalCardId = data.get("globalCardId")
    agentIds = data.get("agentIds")
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: summarize between agents on env {globalCardId}")
    agents = [session[client_sid].registedAgents[agentId] for agentId in agentIds]
    session[client_sid].registedAgents[agentIds[0]].areport(agents, globalCardId, session[client_sid])
    return Response(status=200)

@app.route("/pauseAgentWorkingOnEnvCard", methods=['POST']) 
def pauseAgentWorkingOnEnvCard():
    data = request.get_json()
    client_sid = data.get("sid")
    agentId = data.get("agentId")
    cardId = data.get("cardId")
    if session.get(client_sid) is None or (agentId not in session[client_sid].registedAgents):
        return Response(status=404)
    print(f"logging: pause agent working on env card for card {cardId}")
    session[client_sid].registedAgents[agentId].pause_workflow()
    return Response(status=200)

    
@app.route('/humanInstructOnEnvPaper', methods=['POST'])
def humanInstructOnEnvPaper():
    data = request.get_json()
    ##  agentId, envCardId, fromPaperId, toPaperId, sid: agentServerCommunicator.socketAppId()
    client_sid = data.get("sid")
    agentId = data.get("agentId")
    envCardId = data.get("envCardId")
    fromPaperId = data.get("fromPaperId")
    print(fromPaperId)
    toPaperId = data.get("toPaperId")
    print(toPaperId)
    if session.get(client_sid) is None:
        return Response(status=404)
    print(f"logging: human instruction to read paper: {fromPaperId} and {toPaperId}") 
    session[client_sid].registedAgents[agentId].human_instruction_on_paper(fromPaperId, toPaperId, session[client_sid])
    return Response(status=200)
    



# @app.route('/startAgentsWorkingOnEnvCard', methods=['POST'])
# def startAgentWorkingOnEnvCard():
#     data = request.get_json()
#     client_sid = data.get("sid")
#     cardId = data.get("cardId")
#     agentId = data.get("agentId")
#     task = data.get("task") 
#     if session.get(client_sid) is None or (cardId not in session[client_sid].registedEnvs):
#         return Response(status=404)
#     print(f"logging: start agent working on env card for card {cardId}")
#     session[client_sid].registedAgents[agentId].assign_task(task=task, environment=session[client_sid].registedEnvs[cardId], workspace=None, client_handler=session[client_sid])
#     return Response(status=200)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)