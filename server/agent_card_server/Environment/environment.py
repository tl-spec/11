from agent_card_server.Projection.circularsom import CircularSom, _build_iteration_indexes, get_best_number_of_layers, get_grid_position_som, generate_rr_projection
from agent_card_server.Environment.circular_config import EnvConfig
from agent_card_server.cache_management.manage import CacheManager
import numpy as np
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from pydantic import BaseModel
from agent_card_server.Agent.base import BaseAgent
from agent_card_server.Action.summarize_between_agents import SummarizeAction
from agent_card_server.Environment.faiss import FAISSE
from agent_card_server.Memory.base import MemoryBank
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Any, Dict, Tuple, Union
from sklearn.manifold import TSNE
import asyncio
import uuid

class BaseEnvironment(BaseModel):
    agents: List[BaseAgent] = [] 
    supervisor: BaseAgent = None # role of human
    global_memory_bank: MemoryBank = None

    class Config: 
        arbitrary_types_allowed = True  

    def registerAgent(self, agent: BaseAgent): 
        """"""
        # raise NotImplementedError
        if agent not in self.agents:
            self.agents.append(agent)
            return True

    def unregisterAgent(self, agent: BaseAgent): 
        """"""
        # raise NotImplementedError
        if agent in self.agents:
            self.agents.remove(agent)
            return True

    def interact(self, agent: BaseAgent, action: str, broadcast: bool = False) -> str: 
        """"""
        raise NotImplementedError

    def broadcast(self, agent: BaseAgent, action: str): 
        """"""
        raise NotImplementedError

    def supervise_on_agents(self, agents: List[BaseAgent], action: str): 
        """directly act on a specific agent"""
        raise NotImplementedError    

class ProjectedWorkSpace(BaseModel):
    """Projected workspace"""
    som: Union[CircularSom, Any]
    query: Union[str, Any]
    relevance: Union[List[float], Any]
    computation_id: Union[str, Any] = None
    projected_position: Union[Dict[str, List[float]], Any] # {docstore_id: (x, y)}
    class Config:
        arbitrary_types_allowed = True

    def get_nearby_documents(self, docstore_id: str, visited_paperids: List[str], num: int = 3) -> List[str]:
        ## TODO 1: get the nearby documents based on the distance
        loc_target_doc = np.array(self.projected_position[docstore_id]) 
        all_docs_loc = np.array(list(self.projected_position.values()))
        distances = np.linalg.norm(all_docs_loc - loc_target_doc, axis=1)
        sorted_indices = np.argsort(distances)
        # filter out visited papers and itself
        filtered_indices = [idx for idx in (np.array(list(self.projected_position.keys()))[sorted_indices]).tolist() if (docstore_id != idx and idx not in visited_paperids)][:num]
        return filtered_indices

    def get_inital_documents(self, num: int = 3) -> List[str]:
        loc_target_doc = np.array([0., 0.]) 
        all_docs_loc = np.array(list(self.projected_position.values()))
        distances = np.linalg.norm(all_docs_loc - loc_target_doc, axis=1)
        sorted_indices = np.argsort(distances)
        ## return nearby document ids except the target doc
        return [id for id in (np.array(list(self.projected_position.keys()))[sorted_indices]).tolist()][:num]
    
    def __str__(self): 
        loc_target_doc = np.array([0., 0.])
        
        return f"ProjectedWorkSpace with {len(self.projected_position)} documents"

    def __repr__(self): 
        return f"ProjectedWorkSpace with {len(self.projected_position)} documents"

class InformationSeekingEnvironment(BaseEnvironment): 
    literature_bank: FAISSE = None
    projected_workspace: ProjectedWorkSpace = None
    tsne_projected_workspace: Any = None 
    clientCardId: str = None
    computationId: str = None 
    envName: str = None 
    envId: str = None
    task: str = None
    summary_between_agents: Any = None
    sub_workspaces: Dict[str, Any] = {}#List[InformationSeekingEnvironment | Any] # any refers to a projected radial workspace
    class Config: 
        arbitrary_types_allowed = True

    def create_sub_env(self, sourceEnv: BaseEnvironment, documentIds: List[str], subEnvName:str, cardId: str): 
        """Create a sub environment from the passed in document ids"""
        self.envName = subEnvName 
        self.registerClientCardId(cardId)
        if sourceEnv.task is not None: 
            self.task = sourceEnv.task
        embedding_func = OpenAIEmbeddings() 
        texts = [] 
        embeddings = [] 
        metadata = [] 
        for idx in documentIds: 
            target_doc = sourceEnv.literature_bank.docstore._dict[idx]
            texts.append(target_doc.page_content)
            embeddings.append(target_doc.embedding)
            metadata.append(target_doc.metadata)
        literature_bank = FAISSE.from_embeddings(list(zip(texts, embeddings)), OpenAIEmbeddings(), metadata, documentIds)
        self.literature_bank = literature_bank
        sourceEnv.sub_workspaces[cardId] = self        
        self.project_workspace_with_tsne()
        self.projected_workspace = ProjectedWorkSpace(som=None, query=None, relevance=None, computation_id=None, projected_position={})

        for id, pos in sourceEnv.projected_workspace.projected_position.items():
            if id in documentIds:
                self.projected_workspace.projected_position[id] = pos
        return self.output_env_info()

    def relayoutProjection(self, task: str, config: Dict = None, clientHandler: Any = None, cardId: str = None):
        """Relayout the projection"""
        if cardId is None:
            cardId = self.clientCardId
        self.task = task
        self.project_workspace_with_task(task, config, clientHandler, cardId, save_computation=False)
        return self.output_env_info()
    
    def registerClientCardId(self, cardId: str): 
        self.clientCardId = cardId

    def __str__(self): 
        return f"InformationSeekingEnvironment with {len(self.agents)} agents"

    def load_existing_env(self, path, envName, envId):
        """Load existing environment"""
        self.envName = envName
        self.envId = envId 
        embedding_func = OpenAIEmbeddings() 
        try:
            literature_bank = FAISSE.load_local(path, embedding_func, allow_dangerous_deserialization=True)
        except Exception as e:
            print("check error")
            print(e)
            literature_bank = FAISSE.load_local(path, embedding_func)
        self.literature_bank = literature_bank 
        # TODO: check if the layout is already projected, if so, load it
        self.project_workspace_with_tsne()
        return self.output_env_info(True)
        
    def assign_workspace_with_task(self, task: str, config: Dict = None, clientHandler: Any = None, cardId: str = None):
        """Assign a workspace to the environment"""
        if cardId is None:
            cardId = self.clientCardId
        self.project_workspace_with_task(task, config, clientHandler, cardId)
        return self.output_env_info()
    
    def assign_sub_workspace(self, agent: BaseAgent, sub_workspace: Any):
        """assign a sub workspace to an agent, only id of the documents"""
        agent.workspace = sub_workspace
        self.sub_workspaces[agent.agent_id] = sub_workspace
        # raise NotImplementedError

    def project_workspace_with_tsne(self): 
        """Project the workspace with tsne"""
        if self.literature_bank is None:
            raise ValueError("The literature bank is empty")
        else:
            print("logging: start tsne projection")
            literature_bank = self.literature_bank
            docstore_ids = literature_bank.index_to_docstore_id.values()
            embeddings = np.array([literature_bank.docstore._dict[idx].embedding for idx in docstore_ids])
            data_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity = max(min(20, int(embeddings.shape[0]/5)), 1)).fit_transform(embeddings)
            self.tsne_projected_workspace = {idx: data_embedded[i].tolist() for i, idx in enumerate(docstore_ids)}

    def summarize_between_agents(self, client_handler, globalCardId): 
        """Summarize the information between agents"""
        print("num of agents: ", len(self.agents))
       
        try:
            # in jupyter env
            # Use the current running loop instead of creating a new one.
            asyncio.create_task(self.asummarize_action(client_handler, globalCardId))
            # self.async_start_new_task()
        except RuntimeError as e:
            print("Try Normal Env")
            # in normal python env
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try: 
                # loop.run_until_complete(asyncio.create_task(self.async_start_new_task()))
                loop.run_until_complete(self.asummarize_action(client_handler, globalCardId))
            finally:
                loop.close()

    async def asummarize_action(self, client_handler, globalCardId): 
        """Summarize the action"""
        memories = []
        for agent in self.agents:
            memories.extend(agent.memory.get_inner_memories())
        print("num of inner memories: ", len(memories))
        observation = await SummarizeAction().act(
            query=self.task,
            summaries=memories,
            client_handler=client_handler,
            env=self,
            globalCardId=globalCardId
        )

    def project_workspace_with_task(self, task: str, config: Dict = None, clientHandler: Any = None, cardId: str = None, verbose: bool = False, save_computation=True):
        """
        Project the workspace, only for IR task
        TODO: consider to project the whole space or the subspaces for each agent
        """

        self.task = task 
        if self.literature_bank is None:
            raise ValueError("The literature bank is empty")

        cacheManager = CacheManager() 
        if cacheManager.check_computation_exists(self.envId, task):
            print("logging: load existing computation")
            self.projected_workspace = cacheManager.get_computations_if_available(self.envId, task)
            self.computationId = self.projected_workspace.computation_id
            return self.projected_workspace

        literature_bank = self.literature_bank 
        relevance_pair = literature_bank.similarity_search(task, k=len(literature_bank.docstore._dict), for_task=True) 
        scores = [1 /score for doc, score in relevance_pair]
        scores = scores / np.max(scores)
        _relevance = np.exp(scores)
        relevance = (_relevance - np.min(_relevance))/(np.max(_relevance) - np.min(_relevance))

        embedding_docs = [doc.embedding for doc, score in relevance_pair]
        metadata_docs = [doc.metadata for doc, score in relevance_pair]
        ids_same_order = [doc.docstore_id for doc, score in relevance_pair]
        scaled_embedding_docs = scale(embedding_docs)
        data_size = scaled_embedding_docs.shape[0] # 
        env_config = EnvConfig(config)
        if data_size > 1000:
            env_config.num_of_epochs = 1
        num_of_layers = get_best_number_of_layers(env_config.step, data_size)
        embedding_size = scaled_embedding_docs.shape[1]
        som = CircularSom(env_config.step,  num_of_layers, embedding_size, sigma=env_config.sigma, learning_rate=env_config.learning_rate, activation_distance=env_config.activation_distance, 
                  topology=env_config.topology, neighborhood_function=env_config.neighborhood_function, random_seed=env_config.random_seed)
        if verbose:
            print("logging: start som training")
        som.train(scaled_embedding_docs, relevance, data_size*env_config.num_of_epochs, env_config.w_s, env_config.w_r, client_handler=clientHandler, cardId=cardId, verbose=True)         
        if verbose: 
            print("logging: som training finished")
        circle_pos = get_grid_position_som(som, scaled_embedding_docs, relevance, ids_same_order)
        self.computationId = str(uuid.uuid4())
        self.projected_workspace = ProjectedWorkSpace(som=som, query=task, relevance=relevance, computation_id=self.computationId, projected_position=circle_pos)
        if save_computation:
            cacheManager.save_available_computations(self.envId, task, self.computationId, self.projected_workspace)
        return self.projected_workspace

    def retrieve_candidate_documents_from_workspace(self, agent: BaseAgent, action: str=None) -> List[Document]:
        """
        Get candidate documents for an agent based on its current workspace
        step 1. find agent' current working progress, aka the document read latest 
        step 2. find the near by documents in its reception field 
        step 3. return back to agent the documents that agent has not read yet
        """
        valid_docs_in_reception_field = self.get_agent_reception_field(agent)
        return valid_docs_in_reception_field


    def agent_retrieve_handler(agent: BaseAgent, action: str): 
        """
        step 1. find agent' current working progress 
        step 2. check if agent wants to continue workload or issue new qbr
        step 4. logging the proposed space. 
        step 3. return back to agent documents in its reception field 
        """
        raise NotImplementedError

    def get_agent_reception_field(self, agent: BaseAgent):
        """
        Get the reception field for an agent
        """
        # TODO: implement the real one based on the projected workspace
        workspace = self.literature_bank.docstore._dict.values()
        agent_path = agent.papers_visited
        # simulate one for testing 
        num = 8
        simulated_reception_field = []
        for doc in workspace:
            if doc not in agent_path:
                simulated_reception_field.append(doc)
                num -= 1
            if num == 0:
                break
        return simxulated_reception_field 

    def get_real_agent_reception_field(self, thread):
        """
        Get the reception field for an agent
        """
        # TODO: implement the real one based on the projected workspace

        if len(thread.graph_state.paper_visited_in_commits) == 0:
            docIds = self.projected_workspace.get_inital_documents()
            return [self.literature_bank.docstore._dict[docId] for docId in docIds]
        else: 
            current_position = thread.branch.commits[-1].blobs[0].result['agent_current_position']
            if not current_position or (current_position not in self.projected_workspace.projected_position):
                docIds = self.projected_workspace.get_inital_documents()
                return [self.literature_bank.docstore._dict[docId] for docId in docIds]
            else:
                docIds = self.projected_workspace.get_nearby_documents(current_position, thread.graph_state.visited_paperids)
                return [self.literature_bank.docstore._dict[docId] for docId in docIds]
        

    def output_env_info(self, returnExistingComps: bool = False): 
        """Output environment information
        
        Returns: 
        {
            "documents": {"docId": {documentObj}}, 
            "tsne_projected_workspace": {"docId": [x, y]},
            "task_projected_workspace": {"docId": [x, y]},
            "available_computations": [list of available computations] if envName is provided
            "computation_id": str
        } 
        """
        if self.literature_bank is None:
            return None 
        else: 
            output = {} 
            output["documents"] = {docId: {"title": doc.metadata.get("TI") or "None", "abstract": doc.metadata.get("AB") or "None", "authors": doc.metadata.get("AU") or []} \
                                for (docId, doc) in self.literature_bank.docstore._dict.items()}
            if self.tsne_projected_workspace: 
                output["tsne_projected_workspace"] = self.tsne_projected_workspace 
            
            if self.projected_workspace and self.projected_workspace.projected_position:
                output["task_projected_workspace"] = {
                    "task": self.projected_workspace.query, 
                    "pos": self.projected_workspace.projected_position}

            if returnExistingComps and self.envName: 
                cacheManager = CacheManager()
                output["available_computations"] = cacheManager.get_available_computations_for_env(self.envId)

            if self.projected_workspace and self.projected_workspace.computation_id: 
                output["computation_id"] = self.projected_workspace.computation_id
            
            
        return output
            


        

    
        