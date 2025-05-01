

import importlib
def abm_construct(agent_name, P): 
    return importlib.import_module(f"abm_agent_{agent_name}").abm_agent(P)      # Call the agent() function from the module
