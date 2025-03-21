from functools import wraps
from typing import Callable, Optional, Any, TypeVar, cast, overload
import inspect

F = TypeVar('F', bound=Callable[..., Any])

# First overload: when used as @service
@overload
def service(func: F) -> F: ...

# Second overload: when used as @service() or @service("name")
@overload
def service(*, name: Optional[str] = None) -> Callable[[F], F]: ...

def service(func: Optional[F] = None, *, name: Optional[str] = None) -> Any:
    """
    Decorator to mark a plugin method as an RPC service.

    Can be used as @service or @service(name="custom_name")

    Args:
        func: The function to decorate (when used as @service)
        name: Optional custom name for the service (when used as @service(name="custom_name")).
              If not provided, the function name is used.

    Returns:
        The decorated function with service metadata attached.
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        setattr(wrapper, '_is_service', True)
        setattr(wrapper, '_service_name', name or fn.__name__)
        setattr(wrapper, '_is_async', inspect.iscoroutinefunction(fn))
        return cast(F, wrapper)

    # Handle both @service and @service(name="...") syntaxes
    if func is not None:
        return decorator(func)
    return decorator

