# routing/dispatcher/base.py

class BaseDispatcher:
    async def dispatch(self, backend: str, data: dict, program_id: str = None, call_id: str = None) -> dict:
        raise NotImplementedError
