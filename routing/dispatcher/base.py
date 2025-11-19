# routing/dispatcher/base.py

class BaseDispatcher:
    async def dispatch(self, backend: str, data: dict) -> dict:
        raise NotImplementedError
