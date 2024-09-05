from typing import List, Any, Tuple, Dict
from uuid import UUID


def calculate_synchronized_params(instance: Any) -> List[str]:
    """
    Calculate the list of synchronized parameters for an instance.
    
    Args:
    instance: The object instance to calculate parameters for.
    
    Returns:
    List[str]: List of parameter names that should be synchronized.
    """
    return [
        param for param, value in instance.__dict__.items()
        if isinstance(value, (int, float, str, bool, tuple, UUID)) or 
           (param.endswith('_ID') and value is None)
    ]

def update_state_params(instance: Any, state: Dict[str, Any], recalculate: bool = False) -> None:
    """
    Update the state dictionary with synchronized parameters from the instance.
    
    Args:
    instance: The object instance to synchronize from.
    state: The current state dictionary to update.
    recalculate: Whether to recalculate synchronized parameters.
    """
    if recalculate or not hasattr(instance, 'synchronized_params'):
        instance.synchronized_params = calculate_synchronized_params(instance)
        instance.param_count = len(instance.synchronized_params)

    for param in instance.synchronized_params:
        value = getattr(instance, param)
        if param not in state or state[param] != value:
            state[param] = value

def synchronize_new_parameter(instance: Any, param_name: str) -> None:
    """
    Add a new parameter to the list of synchronized parameters.
    
    Args:
    instance: The object instance to update.
    param_name: The name of the new parameter to synchronize.
    """
    if not hasattr(instance, 'synchronized_params'):
        instance.synchronized_params = calculate_synchronized_params(instance)
        instance.param_count = len(instance.synchronized_params)
    
    if param_name not in instance.synchronized_params:
        instance.synchronized_params.append(param_name)
        instance.param_count += 1
