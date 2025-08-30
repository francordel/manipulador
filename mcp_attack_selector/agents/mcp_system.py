"""
Simulation of an MCP system.
"""
from . import agents

class System():
	"""
	The class to manage the MCP scenario and its entities.
	"""
	def __init__(self,
			 agent: agents.FunctionCallingAgent,
			 system_prompt: str,
			 query: str,
			 tools: list[dict]):
		self._agent = agent
		self._system_prompt = system_prompt
		self._query = query
		self._tools = tools

	def start_pipeline(self):
		"""
		Starts the pipeline of querying the moodel with a query, a system prompt and
		tools the model has access to.


		Returns:
			dspy.Predictio: The LLM's prediction for the given query.
		"""
		return self._agent.run(self._query, self._system_prompt, self._tools)
